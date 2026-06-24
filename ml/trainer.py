from __future__ import annotations

"""
trainer.py - 관심종목 전체 XGBoost 모델 일괄 학습

사용법:
  python -m ml.trainer                   # 전체 STOCKS 학습
  python -m ml.trainer 005930.KS AAPL   # 특정 종목만 학습

학습 데이터: 최근 5년치 일봉 (yfinance)
저장 위치:   ml/models/{ticker}.pkl
"""

import sys
import logging
import yfinance as yf
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _fetch(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"{ticker} 데이터 없음")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    logger.info("  %s: %d행 (%s ~ %s)",
                ticker, len(df), df.index[0].date(), df.index[-1].date())
    return df


def fetch_10y(ticker: str) -> pd.DataFrame:
    """10년치 일봉 데이터 다운로드."""
    logger.info("데이터 다운로드: %s (10년)", ticker)
    return _fetch(ticker, "10y")


def fetch_5y(ticker: str) -> pd.DataFrame:
    """5년치 일봉 데이터 다운로드 (일일 재학습용)."""
    logger.info("데이터 다운로드: %s (5년)", ticker)
    return _fetch(ticker, "5y")


def retrain_daily(market: str = "all", period: str = "5y") -> dict:
    """
    KR reversion: 전체 종목 합산 단일 전역 모델 (train_global) — 백테스트와 동일한 구조.
    trend: 규칙 기반 운용 (ML 학습 없음).
    US: 기존 종목별 train() 방식 유지.
    반환: {"_global_reversion": metrics, ...US tickers...}
    """
    from ml.model import train, train_global
    from ml.features import add_features, _triple_barrier_pnl, FEATURE_COLS, detect_reversion_rows
    from config import TP_PCT, SL_PCT, EOD_HORIZON as HORIZON

    tickers_dict: dict = {}
    kr_tickers: list = []

    if market in ("kr", "all"):
        try:
            from signals.krx_universe import get_krx_backtest_universe
            kr = get_krx_backtest_universe(top_n=200)
            if kr:
                tickers_dict.update(kr)
                kr_tickers = [t for t in kr if t.endswith(".KS") or t.endswith(".KQ")]
                logger.info("KRX 유니버스: %d개", len(kr))
            else:
                raise ValueError("KRX 유니버스 0개")
        except Exception as e:
            logger.warning("KRX 유니버스 조회 실패(%s) → config.STOCKS fallback", e)
            from config import STOCKS
            tickers_dict.update(STOCKS)
            kr_tickers = list(STOCKS.keys())
            logger.info("KRX fallback: config.STOCKS %d개", len(STOCKS))

    if market in ("us", "all"):
        try:
            from signals.us_universe import get_us_backtest_universe
            us_before = len(tickers_dict)
            us = get_us_backtest_universe(top_n=100)
            if us:
                tickers_dict.update(us)
                logger.info("US 유니버스: %d개", len(tickers_dict) - us_before)
            else:
                raise ValueError("US 유니버스 0개")
        except Exception as e:
            logger.warning("US 유니버스 조회 실패(%s) → config.US_STOCKS fallback", e)
            from config import US_STOCKS
            tickers_dict.update(US_STOCKS)
            logger.info("US fallback: config.US_STOCKS %d개", len(US_STOCKS))

    tickers = list(tickers_dict.keys())
    logger.info("재학습 시작: %d개 종목 (market=%s)", len(tickers), market)

    # ── 1단계: 데이터 다운로드 (직렬) ──────────────────────────────
    fetch_fn = fetch_10y if period == "10y" else fetch_5y
    logger.info("1단계: 데이터 다운로드 (직렬, %s)", period)
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            data[ticker] = fetch_fn(ticker)
        except Exception as e:
            logger.error("  [FAIL 데이터] %s: %s", ticker, e)
    logger.info("다운로드 완료: %d / %d개", len(data), len(tickers))

    # ── KOSPI 기준지수 다운로드 ────────────────────────────────────
    try:
        kospi_df_shared = _fetch("^KS11", period)
        logger.info("KOSPI 기준지수 다운로드 완료: %d행", len(kospi_df_shared))
    except Exception as e:
        logger.warning("KOSPI 다운로드 실패(%s) → kospi_relative_5d=NaN", e)
        kospi_df_shared = None

    results: dict = {}

    # ── 2단계: KR reversion 합산 단일 전역 모델 ──────────────────────
    if market in ("kr", "all") and kr_tickers:
        logger.info("2단계: KR reversion 합산 데이터 구성 (%d개 종목)...", len(kr_tickers))
        rev_dfs = []
        for ticker in kr_tickers:
            raw_df = data.get(ticker)
            if raw_df is None:
                continue
            try:
                df = add_features(raw_df, kospi_df=kospi_df_shared)
                labels, future_returns = _triple_barrier_pnl(
                    df, tp_pct=TP_PCT, sl_pct=SL_PCT, max_holding_days=HORIZON
                )
                df = df.copy()
                df["_label"]         = labels
                df["_future_return"] = future_returns
                idx = df.index
                df["_date"]   = idx.tz_localize(None) if getattr(idx, "tzinfo", None) else idx
                df["_ticker"] = ticker
                df = df.dropna(subset=FEATURE_COLS + ["_label", "_future_return"])
                if len(df) > HORIZON:
                    df = df.iloc[:-HORIZON]
                mask = detect_reversion_rows(df).reindex(df.index).fillna(False)
                df   = df[mask]
                if len(df) >= 10:
                    rev_dfs.append(df)
            except Exception as e:
                logger.warning("  [SKIP] %s reversion 피처 실패: %s", ticker, e)

        if rev_dfs:
            combined_rev = pd.concat(rev_dfs).sort_values("_date").reset_index(drop=True)
            logger.info("  reversion 합산: %d행 / %d개 종목", len(combined_rev), len(rev_dfs))
            try:
                _, global_metrics = train_global(combined_rev, agent="reversion")
                results["_global_reversion"] = global_metrics
                logger.info("  KR reversion 전역 모델 완료 auc=%.4f acc=%.3f",
                            global_metrics["auc"], global_metrics["accuracy"])
            except Exception as e:
                logger.error("  KR reversion 전역 모델 학습 실패: %s", e)
                results["_global_reversion"] = None
        else:
            logger.error("reversion 합산 데이터 없음 — 전역 모델 학습 건너뜀")
            results["_global_reversion"] = None

    # ── 3단계: US 종목별 학습 (기존 방식 유지) ───────────────────────
    if market in ("us", "all"):
        from concurrent.futures import ThreadPoolExecutor
        us_tickers = [t for t in tickers if not (t.endswith(".KS") or t.endswith(".KQ"))]
        if us_tickers:
            logger.info("3단계: US 종목별 재학습 (%d개)...", len(us_tickers))

            def _train_us(ticker: str):
                df = data.get(ticker)
                if df is None:
                    return ticker, None
                best = None
                for agent in ("momentum", "reversion"):
                    try:
                        _, m = train(df, ticker, agent=agent)
                        if best is None or m["auc"] > best["auc"]:
                            best = m
                    except Exception as e:
                        logger.warning("  [SKIP] %s [%s]: %s", ticker, agent, e)
                return ticker, best

            with ThreadPoolExecutor(max_workers=8) as pool:
                for ticker, metrics in pool.map(_train_us, us_tickers):
                    results[ticker] = metrics

    ok   = sum(1 for v in results.values() if v)
    fail = len(results) - ok
    logger.info("재학습 완료: 성공 %d / 실패 %d", ok, fail)
    return results


def train_ticker(ticker: str):
    """단일 종목 학습. 성공 시 metrics 반환, 실패 시 None."""
    from ml.model import train
    try:
        df = fetch_5y(ticker)
        _, metrics = train(df, ticker)
        return metrics
    except Exception as e:
        logger.error("[%s] 학습 실패: %s", ticker, e)
        return None


def train_all(tickers: list[str]) -> dict:
    """
    종목 리스트 전체 학습.
    반환: {ticker: metrics or None}
    """
    results = {}
    for ticker in tickers:
        logger.info("=" * 50)
        logger.info("학습 시작: %s", ticker)
        results[ticker] = train_ticker(ticker)
    return results


def print_summary(results: dict):
    """학습 결과 요약 출력."""
    print("\n" + "=" * 60)
    print("학습 결과 요약")
    print("=" * 60)
    success = {k: v for k, v in results.items() if v}
    failed  = [k for k, v in results.items() if not v]

    for ticker, m in success.items():
        print(f"  ✅ {ticker:20s} | acc={m['accuracy']:.3f} | auc={m['auc']:.3f} "
              f"| avg_win={m['avg_win']*100:.1f}% | avg_loss={m['avg_loss']*100:.1f}% "
              f"| N={m['n_samples']}")
    for ticker in failed:
        print(f"  ❌ {ticker:20s} | 학습 실패")

    print(f"\n성공: {len(success)}개 / 실패: {len(failed)}개")


_SECTOR_MAP_CACHE_PATH = "ml/models/sector_map.csv"


def _get_sector_map(tickers: list[str] | None = None) -> dict[str, str]:
    """
    KIS API (FHKST01010100) 로 KR 종목 섹터 매핑. {ticker_with_suffix: sector_name}

    1. ml/models/sector_map.csv 캐시가 있으면 즉시 반환
    2. 없으면 tickers 목록에 대해 KIS API bstp_kor_isnm 조회
    3. 성공 시 CSV로 저장
    4. 조회된 종목이 없으면 빈 dict 반환
    """
    import os, csv, time, requests

    # ── 캐시 로드 ─────────────────────────────────────────────────────────────
    if os.path.exists(_SECTOR_MAP_CACHE_PATH):
        result: dict[str, str] = {}
        with open(_SECTOR_MAP_CACHE_PATH, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) == 2:
                    result[row[0]] = row[1]
        if result:
            logger.info("섹터 매핑 캐시 로드: %d개 종목", len(result))
            return result

    if not tickers:
        logger.warning("섹터 조회 대상 tickers 없음")
        return {}

    # ── KIS API 조회 ──────────────────────────────────────────────────────────
    try:
        from trader import KISTrader
        kis = KISTrader()
    except Exception as e:
        logger.error("KISTrader 초기화 실패: %s → 섹터 데이터 없음", e)
        return {}

    kr_tickers = [t for t in tickers if t.endswith((".KS", ".KQ"))]
    logger.info("KIS API 섹터 조회: %d개 KR 종목", len(kr_tickers))

    result = {}
    url = f"{kis.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"

    for i, ticker in enumerate(kr_tickers):
        code = ticker.replace(".KS", "").replace(".KQ", "")
        try:
            headers = kis._headers("FHKST01010100")
            params  = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code}
            resp    = requests.get(url, headers=headers, params=params, timeout=5)
            output  = resp.json().get("output", {})
            sector  = output.get("bstp_kor_isnm", "").strip()
            if sector:
                result[ticker] = sector
        except Exception as e:
            logger.debug("KIS 섹터 조회 실패 %s: %s", ticker, e)
        time.sleep(0.05)  # KIS API rate limit 대응
        if (i + 1) % 30 == 0:
            logger.info("  섹터 조회 진행: %d / %d", i + 1, len(kr_tickers))

    logger.info("섹터 매핑 완료: %d / %d개 종목", len(result), len(kr_tickers))

    if not result:
        logger.error("KIS API 섹터 조회 결과 0개")
        return {}

    # ── CSV 캐시 저장 ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(_SECTOR_MAP_CACHE_PATH), exist_ok=True)
    with open(_SECTOR_MAP_CACHE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for t, s in result.items():
            writer.writerow([t, s])
    logger.info("섹터 매핑 CSV 저장: %s (%d개)", _SECTOR_MAP_CACHE_PATH, len(result))
    return result


_REVERSION_WF_FOLDS = [
    ("2023-01-01", "2024-01-01", "2025-01-01"),   # train 2023      / valid 2024
    ("2023-01-01", "2025-01-01", "2026-01-01"),   # train 2023~2024 / valid 2025
    ("2023-01-01", "2026-01-01", "2027-01-01"),   # train 2023~2025 / valid 2026
]

_MOMENTUM_WF_FOLDS = _REVERSION_WF_FOLDS  # 동일 3년 walk-forward 구조

# TP=15%, SL=8%, hold=10 (momentum/reversion 공통 최적값)
_LABEL_TP   = 0.15
_LABEL_SL   = 0.08
_LABEL_HOLD = 10


def train_global_models(market: str = "all",
                        agents: tuple[str, ...] = ("momentum", "reversion"),
                        period: str = "5y") -> dict:
    """
    전체 종목 데이터를 합산하여 단일 전역 모델 학습.

    agents: 학습할 에이전트. ("reversion",) 처럼 부분 지정 가능.
    저장: ml/models/_global_momentum.pkl, ml/models/_global_reversion.pkl
    """
    from ml.model import train_global
    from ml.features import (
        add_features, _triple_barrier_pnl,
        detect_momentum_rows, detect_reversion_rows,
        FEATURE_COLS_MOMENTUM, FEATURE_COLS_REVERSION,
    )

    AGENT_FEATURE_COLS = {
        "momentum":  FEATURE_COLS_MOMENTUM,
        "reversion": FEATURE_COLS_REVERSION,
    }

    # dropna 기준: reversion 이 포함되면 reversion 피처 기준 사용
    # 각 에이전트가 독립적으로 dropna 하므로 공통 base 컬럼만 드롭하면 됨
    BASE_COLS = list({c for fc in AGENT_FEATURE_COLS.values() for c in fc
                      if c not in ("sector_relative_5d",)})

    # ── 유니버스 수집 ─────────────────────────────────────────────────────────
    tickers_dict: dict = {}
    if market in ("kr", "all"):
        try:
            from signals.krx_universe import get_krx_backtest_universe
            kr = get_krx_backtest_universe(top_n=200)
            if kr:
                tickers_dict.update(kr)
            else:
                raise ValueError("KRX 유니버스 0개")
        except Exception as e:
            logger.warning("KRX 유니버스 실패(%s) → config.STOCKS fallback", e)
            from config import STOCKS
            tickers_dict.update(STOCKS)

    if market in ("us", "all"):
        try:
            from signals.us_universe import get_us_backtest_universe
            us = get_us_backtest_universe(top_n=100)
            if us:
                tickers_dict.update(us)
            else:
                raise ValueError("US 유니버스 0개")
        except Exception as e:
            logger.warning("US 유니버스 실패(%s) → config.US_STOCKS fallback", e)
            from config import US_STOCKS
            tickers_dict.update(US_STOCKS)

    tickers = list(tickers_dict.keys())
    logger.info("전역 학습 시작: %d개 종목 (market=%s, agents=%s)",
                len(tickers), market, agents)

    # ── KOSPI 데이터 다운로드 ────────────────────────────────────────────────
    logger.info("KOSPI 데이터 다운로드")
    try:
        kospi_df = _fetch("^KS11", "5y")
    except Exception as e:
        logger.warning("KOSPI 다운로드 실패(%s) → kospi_relative_5d=NaN", e)
        kospi_df = None

    # ── 티커 데이터 다운로드 (직렬) ─────────────────────────────────────────
    logger.info("티커 데이터 다운로드 (직렬, %s)", period)
    fetch_fn = fetch_10y if period == "10y" else (
               fetch_5y  if period == "5y"  else
               (lambda t: _fetch(t, period))
    )
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            data[ticker] = fetch_fn(ticker)
        except Exception as e:
            logger.error("  [FAIL 데이터] %s: %s", ticker, e)

    logger.info("다운로드 완료: %d / %d개", len(data), len(tickers))

    # ── 피처 / 라벨 생성 ─────────────────────────────────────────────────────
    logger.info("피처 / 라벨 생성")
    all_rows: dict[str, list] = {a: [] for a in agents}

    for ticker, df in data.items():
        try:
            df_feat = add_features(df, kospi_df=kospi_df)
            labels, future_ret = _triple_barrier_pnl(
                df_feat, tp_pct=_LABEL_TP, sl_pct=_LABEL_SL, max_holding_days=_LABEL_HOLD
            )
            df_feat = df_feat.copy()
            df_feat["_label"]         = labels
            df_feat["_future_return"] = future_ret
            df_feat["_ticker"]        = ticker
            df_feat["_date"]          = df_feat.index
            # base dropna (sector_relative_5d 제외 — 이후 합산 후 계산)
            base_drop = [c for c in BASE_COLS if c in df_feat.columns]
            df_feat = df_feat.dropna(subset=base_drop + ["_label", "_future_return"])
            df_feat = df_feat.iloc[:-_LABEL_HOLD]

            for agent in agents:
                if agent == "momentum":
                    # 전체 데이터로 학습 — 트리거는 예측 시 필터로만 사용
                    sub = df_feat
                else:
                    mask = detect_reversion_rows(df_feat).reindex(df_feat.index).fillna(False)
                    sub  = df_feat[mask]
                if len(sub) >= 5:
                    all_rows[agent].append(sub)
        except Exception as e:
            logger.warning("  [SKIP] %s: %s", ticker, e)

    # ── sector_relative_5d 계산 (reversion 전용, 합산 후) ───────────────────
    if "reversion" in agents and all_rows["reversion"]:
        logger.info("sector_relative_5d 계산 중...")
        sector_map = _get_sector_map(tickers=tickers)
        if not sector_map:
            print("\n❌ pykrx 섹터 조회 실패: sector_relative_5d 를 계산할 수 없습니다.")
            print("   sector_map.csv 캐시도 없음. reversion 학습을 중단합니다.")
            print("   해결: 장이 열린 날 재시도하거나 sector_map.csv 를 수동으로 준비하세요.")
            return {}

        rev_combined = pd.concat(all_rows["reversion"]).sort_values("_date")
        rev_combined["_sector"] = rev_combined["_ticker"].map(sector_map).fillna("unknown")
        n_mapped   = (rev_combined["_sector"] != "unknown").sum()
        n_unknown  = (rev_combined["_sector"] == "unknown").sum()
        logger.info("섹터 매핑 결과: 매핑됨=%d  unknown=%d", n_mapped, n_unknown)

        sect_avg = rev_combined.groupby(
            [pd.Grouper(key="_date", freq="D"), "_sector"]
        )["ret_5d"].transform("mean")
        rev_combined["sector_relative_5d"] = rev_combined["ret_5d"] - sect_avg
        # unknown 섹터(US주식 등) → kospi_relative_5d 로 대체
        mask_unk = rev_combined["_sector"] == "unknown"
        if "kospi_relative_5d" in rev_combined.columns:
            rev_combined.loc[mask_unk, "sector_relative_5d"] = (
                rev_combined.loc[mask_unk, "kospi_relative_5d"]
            )
        else:
            rev_combined.loc[mask_unk, "sector_relative_5d"] = 0.0
        all_rows["reversion"] = [rev_combined]  # 이미 합산됨

    # ── 에이전트별 전역 모델 학습 ─────────────────────────────────────────────
    results = {}
    for agent in agents:
        if not all_rows[agent]:
            logger.warning("[global/%s] 데이터 없음 — 건너뜀", agent)
            continue

        fc = AGENT_FEATURE_COLS[agent]

        if agent == "reversion":
            combined = all_rows["reversion"][0]  # 이미 합산 완료
        else:
            combined = pd.concat(all_rows[agent]).sort_values("_date")

        # reversion: sector_relative_5d dropna 추가
        combined = combined.dropna(subset=fc)

        logger.info("[global/%s] 합산 데이터: %d행 (%d종목)",
                    agent, len(combined), combined["_ticker"].nunique())

        try:
            agent_folds = (_REVERSION_WF_FOLDS if agent == "reversion"
                           else _MOMENTUM_WF_FOLDS if agent == "momentum"
                           else None)
            _, metrics = train_global(combined, agent, feature_cols=fc,
                                      wf_folds=agent_folds)
            results[agent] = metrics
        except Exception as e:
            logger.error("[global/%s] 학습 실패: %s", agent, e)
            results[agent] = None

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("전역 모델 학습 결과")
    print("=" * 65)
    for agent, m in results.items():
        if not m:
            print(f"  ❌ {agent}: 학습 실패")
            continue
        print(f"\n  [{agent}]")
        print(f"  전체 샘플 수  : {m['n_samples']:,}")
        print(f"  pos_rate      : {m['positive_rate']:.3f}")
        print(f"  OOF AUC       : {m['auc']:.4f}  (구 0.5225)")
        if m.get("fold_aucs"):
            print(f"  Fold별 AUC    :")
            for fold_desc, fold_auc in m["fold_aucs"]:
                print(f"    {fold_desc}: {fold_auc:.4f}")
        if m.get("brier_raw") and m.get("brier_cal"):
            print(f"  Brier         : {m['brier_raw']:.4f} → {m['brier_cal']:.4f} (cal)")
        if m.get("oof_proba_stats"):
            s = m["oof_proba_stats"]
            print(f"  OOF win_prob  : min={s['min']:.4f}  mean={s['mean']:.4f}  max={s['max']:.4f}")
            print(f"  OOF p50/p75/p90 : {s['p50']:.4f} / {s['p75']:.4f} / {s['p90']:.4f}")
            print(f"  OOF >= 0.50: {s['above_50']*100:.1f}%  >= 0.52: {s['above_52']*100:.1f}%  >= 0.55: {s['above_55']*100:.1f}%")
        print(f"  피처 중요도 전체 ({len(m.get('feature_importance_all', []))}개):")
        for feat, imp in m.get("feature_importance_all", m.get("feature_importance_top10", [])):
            print(f"    {feat:30s}  {imp:.4f}")

    return results


if __name__ == "__main__":
    # CLI: python -m ml.trainer [--global] [--reversion-only] [--kr] [--3y] [ticker ...]
    if "--global" in sys.argv or len(sys.argv) == 1:
        _agents  = (("reversion",)  if "--reversion-only" in sys.argv else
                    ("momentum",)   if "--momentum-only"  in sys.argv else
                    ("momentum", "reversion"))
        _market  = "kr" if "--kr" in sys.argv else "all"
        _period  = "3y" if "--3y" in sys.argv else "5y"
        train_global_models(market=_market, agents=_agents, period=_period)
    else:
        tickers = [a for a in sys.argv[1:] if not a.startswith("--")]
        logger.info("총 %d개 종목 (종목별) 학습 시작", len(tickers))
        results = train_all(tickers)
        print_summary(results)
