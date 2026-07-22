from __future__ import annotations

"""
kospi200_trainer.py - KOSPI200 XGB 랭킹 전략 전역 모델 재학습

quant_trader_backtest_dev/newstrat 백테스트(HOLD=40/REBAL=40 최적, 전체기간·2025상반기
절단 비교 양쪽에서 코스피 대비 우위 확인)와 동일한 피처·라벨 정의로, 코스피200 PIT
유니버스 전체를 대상으로 XGBoost 회귀 모델을 walk-forward(embargo=HOLD거래일)로 학습한다.
runner.py::retrain_kr_models()가 분기 재학습일에 기존 reversion 재학습 뒤 이어서 호출한다.

사용법: python -m ml.kospi200_trainer
"""

import os
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "models", "kospi200_cache")

UNIV_START_YM = "2015-01"
HOLD = 40  # config.KOSPI200_HOLD과 동일 — 라벨(fwdN)·embargo에 연동

FEATURE_COLS = ["sector_momentum_zscore", "foreigner_institution_flow", "kospi_realized_vol"]


def _build_wf_folds() -> list[tuple]:
    """'오늘' 기준 3-fold expanding walk-forward 경계를 매번 새로 계산한다.

    하드코딩된 캘린더 연도(예: "2025-01-01") 대신 상대 기간을 쓴다 — 이 함수는 40거래일
    리밸런싱마다(scan_kospi200_signals_eod) 재호출되므로, 재학습할 때마다 마지막 fold의
    학습 구간이 자동으로 최신 데이터 쪽으로 밀려 올라간다(고정 연도였다면 시간이 지날수록
    "최근"이 아니게 되어버림).
    """
    today = pd.Timestamp.now().normalize()
    v3 = today.strftime("%Y-%m-%d")
    v2 = (today - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    v1 = (today - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    v0 = (today - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    train_start = f"{UNIV_START_YM}-01"
    return [
        (train_start, v0, v1),
        (train_start, v1, v2),
        (train_start, v2, v3),
    ]


def _ensure_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _is_fresh(df: pd.DataFrame, max_age_days: int = 5) -> bool:
    if df is None or df.empty:
        return False
    last = pd.Timestamp(df.index.max())
    if getattr(last, "tzinfo", None) is not None:
        last = last.tz_localize(None)
    return (pd.Timestamp.now().normalize() - last.normalize()).days <= max_age_days


def _monthly_snapshot_dates(start_ym: str) -> list[str]:
    out = []
    y0, m0 = map(int, start_ym.split("-"))
    today = pd.Timestamp.now()
    y, m = y0, m0
    while (y, m) <= (today.year, today.month):
        out.append(f"{y:04d}-{m:02d}-01")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _cached_pit_snapshot(label: str) -> list[str]:
    from data.kospi200_data import get_kospi200_pit
    _ensure_cache()
    path = os.path.join(CACHE_DIR, f"kospi200_{label.replace('-', '')}.parquet")
    if os.path.exists(path):
        return list(pd.read_parquet(path)["code"])
    codes = get_kospi200_pit(as_of=label)
    pd.DataFrame({"code": codes}).to_parquet(path)
    return codes


def _build_universe() -> tuple[dict[str, list[str]], list[str]]:
    """월별 PIT 스냅샷 dict + 전체기간 등장한 종목코드 풀(중복제거)."""
    snap = {}
    for label in _monthly_snapshot_dates(UNIV_START_YM):
        try:
            snap[label] = _cached_pit_snapshot(label)
        except Exception as e:
            logger.warning("[KOSPI200 재학습] PIT 스냅샷 실패 %s: %s", label, e)
    pool = sorted({c for codes in snap.values() for c in codes})
    return snap, pool


def _universe_membership(dates: pd.Series, codes: pd.Series, snap: dict[str, list[str]]) -> np.ndarray:
    """(date, code) 쌍이 해당 시점 직전 월별 스냅샷에 속했는지 여부(PIT 판정)."""
    import bisect
    snap_dates = sorted(pd.Timestamp(k) for k in snap.keys())
    snap_sets  = {pd.Timestamp(k): set(v) for k, v in snap.items()}
    out = np.zeros(len(dates), dtype=bool)
    for i, (d, c) in enumerate(zip(dates.values, codes.values)):
        idx = bisect.bisect_right(snap_dates, pd.Timestamp(d)) - 1
        if idx >= 0 and c in snap_sets[snap_dates[idx]]:
            out[i] = True
    return out


def _cached_ohlcv(code: str) -> pd.DataFrame:
    from data.data_fetcher import fetch_ohlcv
    _ensure_cache()
    path = os.path.join(CACHE_DIR, f"ohlcv_{code}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if _is_fresh(df):
            return df
    df = fetch_ohlcv(f"{code}.KS", period_years=11)
    df.to_parquet(path)
    return df


def _cached_investor(code: str) -> pd.DataFrame:
    from data.kospi200_data import get_investor_net_buy
    _ensure_cache()
    path = os.path.join(CACHE_DIR, f"investor_{code}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if _is_fresh(df):
            return df
    start = f"{int(UNIV_START_YM[:4]) - 1}-01-01"
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    df = get_investor_net_buy(code, start, end)
    df.to_parquet(path)
    return df


def _build_panel(pool, ohlcv_map, investor_map, kospi_close, sector_map) -> pd.DataFrame:
    rv20 = kospi_close.pct_change().rolling(20).std()

    frames = []
    for c in pool:
        df = ohlcv_map.get(c)
        if df is None or df.empty or len(df) < 60:
            continue
        close = df["Close"].astype(float).sort_index()
        r20  = close / close.shift(20) - 1.0
        fwdN = close.shift(-HOLD) / close - 1.0

        iv = investor_map.get(c)
        if iv is not None and not iv.empty and {"Foreign", "Inst"}.issubset(iv.columns):
            fi = (iv["Foreign"] + iv["Inst"]).reindex(close.index)
        else:
            fi = pd.Series(np.nan, index=close.index)
        amount = (df["Volume"].astype(float) * close).replace(0, np.nan)
        flow_5dma = (fi / amount).rolling(5).mean()

        frames.append(pd.DataFrame({
            "date": close.index, "code": c, "sector": sector_map.get(f"{c}.KS"),
            "r20": r20.values, "fwdN": fwdN.values, "flow": flow_5dma.values,
        }))
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)

    g = panel.groupby(["date", "sector"])["r20"]
    mu    = g.transform("mean")
    sigma = g.transform("std").replace(0, np.nan)
    cnt   = g.transform("count")
    z = (panel["r20"] - mu) / sigma
    panel["sector_momentum_zscore"]     = z.where(cnt >= 3, np.nan)
    panel["kospi_realized_vol"]         = panel["date"].map(rv20)
    panel["foreigner_institution_flow"] = panel["flow"]
    return panel


def _label_panel(panel: pd.DataFrame, snap: dict[str, list[str]]) -> pd.DataFrame:
    in_u = _universe_membership(panel["date"], panel["code"], snap)
    panel = panel[in_u].copy()

    def _pct_rank(s):
        n = len(s)
        if n <= 1:
            return pd.Series(0.5, index=s.index)
        r = s.rank(method="first", ascending=True)
        return (r - 1) / (n - 1)

    lab = panel.dropna(subset=["fwdN"]).groupby("date")["fwdN"]
    panel["_label"] = lab.transform(_pct_rank)
    return panel


def retrain_kospi200_global() -> dict | None:
    """
    KOSPI200 XGB 랭킹 전역 모델 재학습 — 40거래일 리밸런싱 시점마다(scan_kospi200_signals_eod가
    포지션 0개일 때 직접 호출) 최신 데이터까지 포함해 재학습한다. 분기 캘린더가 아니다.

    IC 기반 롤백: 기존 모델을 백업해두고, 새로 학습한 모델과 기존 모델을 "이번 재학습의
    동일한 검증구간"에 나란히 재채점해서 비교한다 — 기존 모델의 IC가 더 높으면 새 모델을
    버리고 기존 모델을 복원한다(reversion의 AUC/avg_win 롤백과 동일한 취지).

    (2026-07-22 수정: 예전에는 "새 모델의 이번 IC" vs "기존 모델의 그때 저장된 IC"를 비교했는데,
    서로 다른 시기(검증연도)를 비교하는 셈이라 결함이 있었다 — quant_trader_backtest_dev
    백테스트로 과거 67개 재학습 시점을 재현해보니, 기존 모델이 "동일 구간"에서도 실제로 더
    나았던 경우는 7.5%(3/40)뿐이었고 나머지 92.5%는 롤백 때문에 더 나은 신규 모델을 계속
    버리고 있었다. 그래서 기존 모델도 이번 검증구간에 다시 채점해 직접 비교하도록 바꿨다.)

    반환 metrics에 old_ic_stale(기존 모델이 학습 당시 기록한 IC, 참고용), old_ic_same_period
    (기존 모델을 이번 검증구간에 재채점한 IC, 실제 롤백 판단 기준), rolled_back(bool) 포함.
    실패 시 None.
    """
    from ml.model import train_global_regression, load_model, _model_path
    from ml.trainer import _get_sector_map
    import shutil

    # 기존 모델 백업 + 재채점용 모델 객체 확보. 최초 학습이면 파일이 없어 old_model=None.
    model_path  = _model_path("_global", "kospi200_xgb")
    backup_path = model_path + ".bak"
    old_model, old_ic_stale = None, None
    if os.path.exists(model_path):
        shutil.copy2(model_path, backup_path)
        old_model, old_metrics = load_model("_global", "kospi200_xgb")
        old_ic_stale = old_metrics.get("ic") if old_metrics else None
        logger.info("[KOSPI200 재학습] 기존 모델 백업 완료 (학습 당시 IC=%s, 참고용)", old_ic_stale)

    logger.info("[KOSPI200 재학습] 시작")
    snap, pool = _build_universe()
    if not pool:
        logger.error("[KOSPI200 재학습] PIT 유니버스 조회 실패 — 중단")
        return None
    logger.info("[KOSPI200 재학습] PIT 유니버스 풀 %d종목 (스냅샷 %d개월)", len(pool), len(snap))

    sector_map = _get_sector_map(tickers=[f"{c}.KS" for c in pool])
    if not sector_map:
        logger.error("[KOSPI200 재학습] 섹터 매핑 실패 — 중단")
        return None

    ohlcv_map, investor_map = {}, {}
    for i, c in enumerate(pool):
        try:
            ohlcv_map[c] = _cached_ohlcv(c)
        except Exception as e:
            logger.warning("[KOSPI200 재학습] OHLCV 실패 %s: %s", c, e)
        try:
            investor_map[c] = _cached_investor(c)
        except Exception as e:
            logger.warning("[KOSPI200 재학습] 투자자데이터 실패 %s: %s", c, e)
        if (i + 1) % 50 == 0:
            logger.info("[KOSPI200 재학습] 데이터수집 진행 %d/%d", i + 1, len(pool))
    logger.info("[KOSPI200 재학습] OHLCV %d개 투자자데이터 %d개 수집 완료", len(ohlcv_map), len(investor_map))

    import yfinance as yf
    kospi_df = yf.download("^KS11", period="15y", auto_adjust=True, progress=False)
    if kospi_df is None or kospi_df.empty:
        logger.error("[KOSPI200 재학습] KOSPI 지수 다운로드 실패 — 중단")
        return None
    if isinstance(kospi_df.columns, pd.MultiIndex):
        kospi_df.columns = kospi_df.columns.get_level_values(0)
    kospi_close = kospi_df["Close"].astype(float)
    if getattr(kospi_close.index, "tz", None) is not None:
        kospi_close.index = kospi_close.index.tz_localize(None)

    panel = _build_panel(pool, ohlcv_map, investor_map, kospi_close, sector_map)
    panel = _label_panel(panel, snap)
    logger.info("[KOSPI200 재학습] PIT 라벨링 후 %d행", len(panel))

    feat = panel.copy()
    feat["sector_momentum_zscore"]     = feat["sector_momentum_zscore"].fillna(0.0)
    feat["foreigner_institution_flow"] = feat["foreigner_institution_flow"].fillna(0.0)
    feat = feat[feat["kospi_realized_vol"].notna() & feat["_label"].notna()].copy()
    feat = feat.rename(columns={"date": "_date"})

    if len(feat) < 1000:
        logger.error("[KOSPI200 재학습] 학습표본 부족(%d행) — 중단", len(feat))
        return None

    try:
        _, metrics, (X_val, y_val) = train_global_regression(
            feat, agent="kospi200_xgb",
            feature_cols=FEATURE_COLS,
            wf_folds=_build_wf_folds(), embargo_days=HOLD,
        )
    except Exception as e:
        logger.error("[KOSPI200 재학습] 모델 학습 실패: %s", e)
        return None

    metrics["old_ic_stale"] = old_ic_stale
    metrics["old_ic_same_period"] = None
    metrics["rolled_back"] = False
    if old_model is not None:
        # 기존 모델을 "이번 재학습의 검증구간(X_val, y_val)"에 재채점 — 동일 기간 비교
        old_pred = old_model.predict(X_val)
        old_ic_same_period = pd.Series(y_val).corr(pd.Series(old_pred), method="spearman")
        old_ic_same_period = float(old_ic_same_period) if pd.notna(old_ic_same_period) else 0.0
        metrics["old_ic_same_period"] = old_ic_same_period

        if old_ic_same_period > metrics["ic"]:
            shutil.copy2(backup_path, model_path)
            metrics["rolled_back"] = True
            logger.warning("[KOSPI200 재학습] 기존 모델이 동일구간에서 더 나음(기존=%.4f > 신규=%.4f, "
                           "기존의 학습당시 stale IC=%s) — 롤백, 기존 모델 유지",
                           old_ic_same_period, metrics["ic"],
                           f"{old_ic_stale:.4f}" if old_ic_stale is not None else "없음")
        else:
            logger.info("[KOSPI200 재학습] 완료 IC=%.4f(동일구간 재채점한 기존=%.4f, oof=%.4f) n=%d "
                       "— 신규 모델 채택", metrics["ic"], old_ic_same_period,
                       metrics["oof_ic"], metrics["n_samples"])
    else:
        logger.info("[KOSPI200 재학습] 완료 IC=%.4f(최초학습, oof=%.4f) n=%d",
                    metrics["ic"], metrics["oof_ic"], metrics["n_samples"])
    return metrics


if __name__ == "__main__":
    result = retrain_kospi200_global()
    if result:
        print("\n" + "=" * 60)
        print("KOSPI200 XGB 랭킹 전역 모델 재학습 결과")
        print("=" * 60)
        print(f"  IC (마지막 fold) : {result['ic']:.4f}")
        print(f"  OOF IC           : {result['oof_ic']:.4f}")
        print(f"  기존 stale IC    : {result.get('old_ic_stale')}")
        print(f"  기존 동일구간 IC : {result.get('old_ic_same_period')}")
        print(f"  롤백 여부        : {result['rolled_back']}")
        print(f"  샘플 수          : {result['n_samples']:,}")
        for fold_label, ic in result["fold_ics"]:
            print(f"  {fold_label}: IC={ic:.4f}")
        print("\n  피처 중요도:")
        for feat_name, imp in result["feature_importance_all"]:
            print(f"    {feat_name:30s} {imp:.4f}")
    else:
        print("재학습 실패 — 로그 확인 필요")
