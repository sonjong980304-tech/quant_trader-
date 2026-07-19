"""
backtest_walkforward.py - Walk-forward ML 백테스트 (비용 반영)

Walk-forward 구조:
  - 학습 창: 2년 일봉 데이터
  - 테스트 창: 3개월
  - 슬라이딩 스텝: 3개월

비용 모델 (왕복):
  - 수수료: 0.015% × 2 (매수+매도) = 0.03%
  - 슬리피지: 0.25% (진입 시 단방향 가정, 왕복 0.25%)
  - 증권거래세: 0.18% (매도 시, KRX 한국 종목만)

Triple-Barrier 청산:
  - TP=+15%, SL=-6%, 최대 7거래일 (G1 그리드 채택, config 단일 진실 소스)
  - 당일 High/Low로 판정 (진입 다음날부터)

게이트:
  - 비용 차감 후 기대값(expectancy) ≤ 0 → 실거래 재개 불가
  - 승률이 높더라도 비용 후 음수면 명확히 보고

실행:
  python backtest_walkforward.py
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo-root: 하위 폴더에서 직접 실행 대비

import logging
import sys
import warnings
from datetime import date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    ML_MIN_WIN_PROB    as MIN_WIN_PROB,
    ML_MIN_RISK_REWARD as MIN_RR,
    TP_PCT,
    SL_PCT,
    EOD_HORIZON,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 비용 모델 상수
# ─────────────────────────────────────────────────────────────────────────────

COMMISSION_PCT = 0.00015   # 한쪽 수수료 0.015% → 왕복 0.03%
SLIPPAGE_PCT   = 0.0005    # 진입 슬리피지 0.05% (익일 시초가 지정가 기준)
STT_PCT        = 0.0018    # 증권거래세 0.18% (한국 종목 매도 시)

# Walk-forward 파라미터
TRAIN_MONTHS   = 24        # 학습 창 (개월)
TEST_MONTHS    = 3         # 테스트 창 (개월)
STEP_MONTHS    = 3         # 슬라이딩 스텝 (개월)

# 신호 필터 (MIN_WIN_PROB / MIN_RR / TP_PCT / SL_PCT → config 단일 진실 소스)
MIN_AUC        = 0.58

# Triple-Barrier
HORIZON        = EOD_HORIZON  # 최대 보유 거래일 (config 단일 진실 소스)
ATR_MULT       = 2.0       # ATR 기반 SL 승수 (2×ATR)


# ─────────────────────────────────────────────────────────────────────────────
# 비용 계산
# ─────────────────────────────────────────────────────────────────────────────

def _apply_costs(raw_pnl_pct: float, is_korean: bool) -> float:
    """거래 비용 차감 후 실질 손익률 반환."""
    stt = STT_PCT if is_korean else 0.0
    total_cost = (COMMISSION_PCT * 2) + SLIPPAGE_PCT + stt
    return raw_pnl_pct - total_cost


def _is_korean_ticker(ticker: str) -> bool:
    return ticker.endswith(".KS") or ticker.endswith(".KQ")


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward 폴드 생성
# ─────────────────────────────────────────────────────────────────────────────

def _make_folds(
    df: pd.DataFrame,
    train_months: int = TRAIN_MONTHS,
    test_months:  int = TEST_MONTHS,
    step_months:  int = STEP_MONTHS,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """(train_idx, test_idx) 튜플 목록 반환. 겹침 없음."""
    if df.empty:
        return []
    min_date = df.index.min()
    max_date = df.index.max()

    folds = []
    test_start = min_date + relativedelta(months=train_months)
    while True:
        test_end = test_start + relativedelta(months=test_months)
        if test_end > max_date:
            break
        train_mask = (df.index >= min_date) & (df.index < test_start)
        test_mask  = (df.index >= test_start) & (df.index < test_end)
        if train_mask.sum() >= 120 and test_mask.sum() >= 10:
            folds.append((df.index[train_mask], df.index[test_mask]))
        test_start += relativedelta(months=step_months)

    return folds


# ─────────────────────────────────────────────────────────────────────────────
# Triple-Barrier 청산 (일봉 High/Low 기반)
# ─────────────────────────────────────────────────────────────────────────────

def _barrier_exit(
    future_df: pd.DataFrame,
    entry_price: float,
    atr: float = 0.0,
) -> tuple[float, str]:
    """
    진입 이후 HORIZON 거래일 이내 배리어 판정.

    Args:
        future_df   : 진입 다음날부터의 OHLCV 슬라이스 (최대 HORIZON행)
        entry_price : 진입 가격
        atr         : ATR14 값 (>0이면 ATR 기반 SL 사용, 0이면 고정 SL_PCT)

    Returns:
        (raw_pnl_pct, reason)  reason ∈ {'tp', 'sl', 'vertical'}
    """
    tp_price = entry_price * (1 + TP_PCT)
    # 고정 SL — 라벨 생성(_triple_barrier_pnl)과 동일한 기준 사용
    # ATR 기반 SL은 라벨과 불일치를 유발하므로 제거 (Phase D)
    sl_price      = entry_price * (1 - SL_PCT)
    actual_sl_pct = SL_PCT
    window        = future_df.iloc[:HORIZON]

    for _, row in window.iterrows():
        hit_sl = float(row["Low"])  <= sl_price
        hit_tp = float(row["High"]) >= tp_price
        if hit_sl:
            return -actual_sl_pct, "sl"
        if hit_tp:
            return TP_PCT, "tp"

    # 시간 배리어: 마지막 행 종가 기준
    final_close = float(window["Close"].iloc[-1]) if not window.empty else entry_price
    raw = (final_close - entry_price) / entry_price
    return raw, "vertical"


# ─────────────────────────────────────────────────────────────────────────────
# 단일 종목 Walk-forward
# ─────────────────────────────────────────────────────────────────────────────

def walkforward_ticker(
    ticker: str,
    df_daily: pd.DataFrame,
    kospi_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    단일 종목의 모든 fold에서 매매를 시뮬레이션. trades 목록 반환.
    kospi_df: KOSPI 일봉 데이터 (kospi_relative 피처 계산용, None이면 NaN 처리)
    """
    from ml.features import add_features, FEATURE_COLS

    folds = _make_folds(df_daily)
    if not folds:
        return []

    is_korean = _is_korean_ticker(ticker)
    trades: list[dict] = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        # ── 피처 계산 (컨텍스트 보존) ─────────────────────────────
        context_df = df_daily[df_daily.index <= test_idx[-1]]
        try:
            feat_df   = add_features(context_df, kospi_df=kospi_df).dropna(subset=FEATURE_COLS)
        except Exception:
            continue
        test_feat = feat_df[feat_df.index.isin(test_idx)]
        if test_feat.empty:
            continue

        # ── 트리거 사전 계산 ──────────────────────────────────────
        trig_map = _precompute_triggers(context_df)

        # ── 신호 탐지 → 매매 시뮬레이션 ──────────────────────────
        in_position_until: pd.Timestamp | None = None

        for signal_date in test_feat.index:
            if in_position_until is not None and signal_date <= in_position_until:
                continue

            # ① 트리거 확인 (없으면 skip — 실거래 정합성)
            triggers = trig_map.get(signal_date, [])
            if not triggers:
                continue

            # ③ 진입: 다음 거래일 시가
            future_all = df_daily[df_daily.index > signal_date]
            if len(future_all) < 2:
                continue

            entry_price = float(future_all["Open"].iloc[0])
            entry_date  = future_all.index[0]

            # ⑤ 청산: Triple-Barrier
            exit_window = future_all.iloc[1 : HORIZON + 1]
            if exit_window.empty:
                continue

            raw_pnl, reason = _barrier_exit(exit_window, entry_price)
            net_pnl         = _apply_costs(raw_pnl, is_korean)

            exit_date  = exit_window.index[min(HORIZON - 1, len(exit_window) - 1)]
            exit_price = (
                entry_price * (1 + TP_PCT)    if reason == "tp"
                else entry_price * (1 - SL_PCT) if reason == "sl"
                else float(exit_window["Close"].iloc[-1])
            )

            in_position_until = exit_date
            trades.append({
                "ticker":      ticker,
                "fold":        fold_i,
                "signal_date": str(signal_date.date()),
                "entry_date":  str(entry_date.date()),
                "exit_date":   str(exit_date.date()),
                "entry_price": round(entry_price, 4),
                "exit_price":  round(exit_price, 4),
                "raw_pnl_pct": round(raw_pnl * 100, 3),
                "net_pnl_pct": round(net_pnl * 100, 3),
                "exit_reason": reason,
                "triggers":    ",".join(triggers),
                "is_win":      int(net_pnl > 0),
            })

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 수집
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_daily(ticker: str, years: int = 5) -> pd.DataFrame:
    end   = date.today()
    start = end - relativedelta(years=years)
    df    = yf.download(ticker, start=str(start), end=str(end),
                        auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 전체 실행 + 결과 보고
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_STOCKS = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "035420.KS": "NAVER",
    "005380.KS": "현대차",
    "051910.KS": "LG화학",
    "006400.KS": "삼성SDI",
    "035720.KS": "카카오",
    "000270.KS": "기아",
    "068270.KS": "셀트리온",
    "028260.KS": "삼성물산",
    "105560.KS": "KB금융",
    "055550.KS": "신한지주",
    "012330.KS": "현대모비스",
    "096770.KS": "SK이노베이션",
    "034730.KS": "SK",
}


def run_walkforward(stocks: dict | None = None) -> str:
    """
    Walk-forward 백테스트 실행.

    Returns:
        결과 요약 문자열 (게이트 통과 여부 포함)
    """
    if stocks is None:
        try:
            from signals.krx_universe import get_krx_backtest_universe
            # use_static=True: 2010년 이전 상장 고정 리스트 → 생존편향 최소화 (GATE A)
            stocks = get_krx_backtest_universe(top_n=50, use_static=True)
        except Exception:
            stocks = {}
        if not stocks:
            from config import STOCKS
            stocks = dict(STOCKS) if STOCKS else {}
        if not stocks:
            stocks = _FALLBACK_STOCKS
            logger.info("KRX universe/config 비어 있음 — fallback 15종목 사용")

    logger.info("=== Walk-forward 백테스트 시작 (%d종목) ===", len(stocks))

    # KOSPI 일봉 데이터 사전 로드 (kospi_relative 피처 계산용)
    _kospi_df: pd.DataFrame | None = None
    try:
        import yfinance as _yf
        _kraw = _yf.download("^KS11", start="2018-01-01", progress=False, auto_adjust=True)
        if hasattr(_kraw.columns, "levels"):
            _kraw.columns = _kraw.columns.get_level_values(0)
        if not _kraw.empty:
            _kospi_df = _kraw[["Close"]].copy()
            logger.info("KOSPI 데이터 로드 완료 (%d행)", len(_kospi_df))
    except Exception as _ke:
        logger.warning("KOSPI 데이터 로드 실패 (kospi_relative NaN 처리): %s", _ke)

    all_trades: list[dict] = []

    for ticker, name in stocks.items():
        try:
            df = _fetch_daily(ticker)
        except Exception as e:
            logger.warning("[%s] 데이터 수집 실패: %s", ticker, e)
            continue

        min_rows = (TRAIN_MONTHS + TEST_MONTHS) * 20 + HORIZON
        if len(df) < min_rows:
            logger.info("[%s] 데이터 부족 (%d행) — 스킵", ticker, len(df))
            continue

        logger.info("[%s] walk-forward 시작...", ticker)
        trades = walkforward_ticker(ticker, df, kospi_df=_kospi_df)
        logger.info("[%s] %d건 완료", ticker, len(trades))
        all_trades.extend(trades)

    return _format_result(all_trades, stocks)


def _format_result(trades: list[dict], stocks: dict) -> str:
    if not trades:
        return (
            "⚠️ Walk-forward 백테스트: 신호 없음\n"
            "→ 엣지 확인 불가 — 실거래 재개 불가"
        )

    df = pd.DataFrame(trades)

    total          = len(df)
    wins           = int(df["is_win"].sum())
    win_rate       = wins / total * 100
    avg_raw        = df["raw_pnl_pct"].mean()
    avg_net        = df["net_pnl_pct"].mean()
    total_cost_pct = avg_raw - avg_net
    expectancy     = avg_net / 100

    # ── GATE C-1: Bootstrap 95% CI (2000회 리샘플링) ───────────────
    pnl_arr = df["net_pnl_pct"].values / 100
    rng     = np.random.default_rng(42)
    boot_ev = [rng.choice(pnl_arr, size=len(pnl_arr), replace=True).mean()
               for _ in range(2000)]
    ci_low  = float(np.percentile(boot_ev, 2.5))
    ci_high = float(np.percentile(boot_ev, 97.5))
    ci_pass = ci_low > 0  # 95% CI 하단이 0 초과여야 진짜 엣지

    # ── GATE C-2: 슬리피지 민감도 ────────────────────────────────
    # raw_pnl에서 비용만 달리 적용 (전체 재실행 없이 재계산)
    slip_rows = []
    for slip_pct in [0.25, 0.40, 0.60, 0.80]:
        slip = slip_pct / 100
        cost = COMMISSION_PCT * 2 + slip + STT_PCT  # 한국 기준
        adj  = df["raw_pnl_pct"] - cost * 100
        ev   = adj.mean() / 100
        slip_rows.append((slip_pct, ev))

    # ── GATE C-3: 종목 집중도 ─────────────────────────────────────
    by_ticker = (
        df.groupby("ticker")
        .agg(
            trades=("is_win", "count"),
            win_rate=("is_win", lambda x: x.mean() * 100),
            avg_net=("net_pnl_pct", "mean"),
        )
        .sort_values("trades", ascending=False)
    )
    top_ticker      = by_ticker.index[0]
    top_conc_pct    = float(by_ticker.iloc[0]["trades"]) / total * 100
    top_name        = stocks.get(top_ticker, top_ticker)
    concentration_warn = (
        f"⚠️ 집중도 경고: {top_name}({top_ticker}) {top_conc_pct:.0f}%"
        if top_conc_pct > 30 else
        f"✅ 집중도 양호: 최대 {top_name} {top_conc_pct:.0f}%"
    )

    # n<10 종목 필터링 경고 (소표본 신뢰도 낮음)
    thin_tickers = by_ticker[by_ticker["trades"] < 10].index.tolist()

    # 청산 유형 분포
    reason_counts = df["exit_reason"].value_counts().to_dict()

    # ── 최종 GATE 판정 ──────────────────────────────────────────────
    # 조건: EV > 0  AND  CI 하단 > 0  AND  집중도 < 50%
    gate_ev   = expectancy > 0
    gate_ci   = ci_pass
    gate_conc = top_conc_pct < 50
    gate_pass = gate_ev and gate_ci and gate_conc

    if gate_pass:
        gate_label = "✅ GATE C 통과 — EV > 0, CI 하단 > 0, 집중도 < 50%"
    else:
        reasons = []
        if not gate_ev:
            reasons.append(f"EV={expectancy:+.4f} ≤ 0")
        if not gate_ci:
            reasons.append(f"CI 하단={ci_low:+.4f} ≤ 0 (엣지 불안정)")
        if not gate_conc:
            reasons.append(f"집중도 {top_conc_pct:.0f}% ≥ 50% ({top_name})")
        gate_label = f"❌ GATE C 미통과 — {' | '.join(reasons)}\n→ 엣지가 소수 종목/비현실적 비용 가정에 의존. 실거래 불가."

    lines = [
        "📊 <b>Walk-forward 백테스트 결과 (GATE C 포함)</b>",
        f"유니버스: {len(stocks)}종목 (정적 장기상장 리스트, 생존편향 최소화)",
        f"총 거래: {total}건 | 승률: {win_rate:.1f}%",
        f"평균 수익(세전): {avg_raw:+.3f}%",
        f"평균 비용: -{total_cost_pct:.3f}%",
        f"평균 수익(세후): {avg_net:+.3f}%",
        f"기대값(세후): {expectancy:+.4f}",
        "",
        f"청산 유형: TP={reason_counts.get('tp',0)} / SL={reason_counts.get('sl',0)} / 기간={reason_counts.get('vertical',0)}",
        "",
        "<b>[GATE C-1] 부트스트랩 95% CI (n=2000)</b>",
        f"  EV CI: [{ci_low:+.4f}, {ci_high:+.4f}]",
        f"  {'✅ CI 하단 > 0 — 엣지 통계적으로 유의' if ci_pass else '❌ CI 하단 ≤ 0 — 엣지 불안정'}",
        "",
        "<b>[GATE C-2] 슬리피지 민감도 (한국 기준)</b>",
    ]

    for slip_pct, ev in slip_rows:
        mark = "✅" if ev > 0 else "❌"
        lines.append(f"  {mark} 슬리피지 {slip_pct:.2f}%: EV={ev:+.4f}")

    lines += [
        "",
        "<b>[GATE C-3] 종목 집중도</b>",
        f"  {concentration_warn}",
    ]

    if thin_tickers:
        lines.append(f"  ⚠️ n<10 소표본 종목 (신뢰도 낮음): {', '.join(thin_tickers)}")

    lines += ["", "<b>종목별 성과 (거래건수 기준)</b>"]
    for ticker, row in by_ticker.iterrows():
        name = stocks.get(ticker, ticker)
        lines.append(
            f"  {name}({ticker}): {int(row['trades'])}건 | "
            f"승률{row['win_rate']:.0f}% | 세후{row['avg_net']:+.2f}%"
        )

    lines += ["", gate_label]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Phase F — ML 기여도 검증 인프라
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_ci(pnl_arr: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """단일 전략 EV의 부트스트랩 95% CI."""
    rng  = np.random.default_rng(seed)
    boot = [rng.choice(pnl_arr, size=len(pnl_arr), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _bootstrap_diff_ci(pnl_a: np.ndarray, pnl_b: np.ndarray,
                       n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """두 전략 EV 차이(a - b)의 부트스트랩 95% CI."""
    rng   = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(pnl_a, size=len(pnl_a), replace=True).mean()
        sb = rng.choice(pnl_b, size=len(pnl_b), replace=True).mean()
        diffs.append(sa - sb)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _strategy_metrics(trades_list: list[dict]) -> dict:
    """전략 요약 지표 산출 (EV·CI·Sharpe·MDD)."""
    if not trades_list:
        return dict(n=0, win_rate=0.0, ev=0.0, ci_low=0.0, ci_high=0.0, sharpe=0.0, mdd=0.0)
    pnl  = np.array([t["net_pnl_pct"] for t in trades_list]) / 100
    n    = len(pnl)
    ev   = float(pnl.mean())
    wr   = float((pnl > 0).mean() * 100)
    ci_l, ci_h = _bootstrap_ci(pnl)
    sharpe = float(ev / pnl.std() * np.sqrt(n)) if pnl.std() > 1e-9 else 0.0
    cum  = np.cumprod(1 + pnl)
    mdd  = float((cum / np.maximum.accumulate(cum) - 1).min() * 100)
    return dict(n=n, win_rate=wr, ev=ev, ci_low=ci_l, ci_high=ci_h, sharpe=sharpe, mdd=mdd)


def _precompute_triggers(df: pd.DataFrame) -> dict:
    """
    OHLCV df의 모든 날짜에 대해 트리거 플래그를 벡터 연산으로 사전 계산.
    detect_triggers()의 5가지 조건을 동일하게 구현 (반복 슬라이싱 대신).

    반환: {Timestamp: [트리거명, ...]}
    """
    if len(df) < 62:
        return {}

    close  = df["Close"].squeeze().astype(float)
    open_  = df["Open"].squeeze().astype(float)
    volume = df["Volume"].squeeze().astype(float)

    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    vol_ma20 = volume.rolling(20).mean()

    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    avg_loss = (-delta).clip(lower=0).ewm(com=13, min_periods=14).mean()
    rsi      = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))

    ema20     = close.ewm(span=20, adjust=False).mean()
    deviation = (close - ema20) / ema20.replace(0, np.nan)
    min_bb60  = bb_width.rolling(60).min().shift(1)

    result: dict = {}
    for i in range(62, len(df)):
        date = df.index[i]
        trig = []
        vm   = float(vol_ma20.iloc[i - 1]) if not np.isnan(float(vol_ma20.iloc[i - 1])) else 0.0
        vi   = float(volume.iloc[i])
        if vm > 0 and vi > vm * 2.0 and float(close.iloc[i]) > float(open_.iloc[i]):
            trig.append("거래량폭발")
        if float(close.iloc[i - 1]) < float(bb_lower.iloc[i - 1]) and float(close.iloc[i]) >= float(bb_lower.iloc[i]):
            trig.append("BB하단반등")
        if float(rsi.iloc[i - 1]) < 30 and float(rsi.iloc[i]) >= 30:
            trig.append("RSI과매도탈출")
        if float(deviation.iloc[i]) <= -0.05:
            trig.append("이격도저점")
        mw = float(min_bb60.iloc[i]) if not np.isnan(float(min_bb60.iloc[i])) else np.nan
        if not np.isnan(mw) and float(bb_width.iloc[i - 1]) <= mw * 1.1 and float(close.iloc[i]) > float(bb_upper.iloc[i]):
            trig.append("BB스퀴즈돌파")
        result[date] = trig
    return result


def _walk_strategy(ticker: str, df_daily: pd.DataFrame, strategy: str,
                   threshold: float | None = None) -> list[dict]:
    """
    단일 종목·단일 전략 walk-forward 시뮬레이션.

    strategy:
      "trigger"  (B1) — 트리거 발생 시 무조건 매수
      "random"   (B0) — 트리거 발생 시 50% 확률 매수
      "rule"     (B2) — 트리거 + Close > MA60
      "ml"            — 트리거 + ML 확률 ≥ MIN_WIN_PROB (기존 모델 재사용)

    비용·청산 모델·유니버스·기간은 GATE C와 완전히 동일.
    """
    import os, pickle
    from ml.features import add_features, FEATURE_COLS

    folds = _make_folds(df_daily)
    if not folds:
        return []

    is_korean = _is_korean_ticker(ticker)
    trades:  list[dict] = []
    rng = np.random.default_rng(42)

    # MA60 (B2용)
    ma60_series = df_daily["Close"].rolling(60).mean()

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        context_df = df_daily[df_daily.index <= test_idx[-1]]

        # 피처 계산 (트리거 날짜 필터링용 + ML용)
        try:
            feat_df   = add_features(context_df).dropna(subset=FEATURE_COLS)
        except Exception:
            continue
        test_feat = feat_df[feat_df.index.isin(test_idx)]
        if test_feat.empty:
            continue

        # 트리거 사전 계산 (B0/B1/B2/ML 공통 — 모든 전략이 트리거를 베이스로 사용)
        trig_map = _precompute_triggers(context_df)

        # ML 모델 로드 (strategy="ml"일 때만)
        model     = None
        avg_win   = TP_PCT
        avg_loss  = SL_PCT
        if strategy == "ml":
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "ml", "models",
                f"{ticker.replace('.', '_')}_wf_f{fold_i}.pkl",
            )
            if os.path.exists(model_path):
                with open(model_path, "rb") as fh:
                    saved     = pickle.load(fh)
                model     = saved["model"]
                avg_win   = saved["metrics"].get("avg_win", TP_PCT)
                avg_loss  = saved["metrics"].get("avg_loss", SL_PCT)
            else:
                # 모델 없으면 이 fold 스킵
                continue

        in_position_until = None

        for signal_date in test_feat.index:
            if in_position_until is not None and signal_date <= in_position_until:
                continue

            # MA200 추세 필터 (동일)
            hist = df_daily[df_daily.index <= signal_date]
            if len(hist) >= 200:
                ma200 = float(hist["Close"].rolling(200).mean().iloc[-1])
                if float(hist["Close"].iloc[-1]) < ma200:
                    continue

            # 트리거 확인 (모든 전략 공통 베이스)
            triggers = trig_map.get(signal_date, [])
            if not triggers:
                continue

            # 전략별 2차 필터
            if strategy == "trigger":
                pass  # 트리거만으로 충분

            elif strategy == "random":
                if rng.random() >= 0.5:
                    continue

            elif strategy == "rule":
                ma60_val = float(ma60_series.get(signal_date, np.nan))
                close_val = float(df_daily.loc[signal_date, "Close"]) if signal_date in df_daily.index else np.nan
                if np.isnan(ma60_val) or np.isnan(close_val) or close_val <= ma60_val:
                    continue

            elif strategy == "ml":
                if model is None:
                    continue
                X  = test_feat.loc[[signal_date], FEATURE_COLS].values.astype("float32")
                wp = float(model.predict_proba(X)[0, 1])
                expected_win  = avg_win  * wp
                expected_loss = avg_loss * (1 - wp)
                rr = expected_win / expected_loss if expected_loss > 0 else 0.0
                min_wp = threshold if threshold is not None else MIN_WIN_PROB
                if wp < min_wp or rr < MIN_RR:
                    continue

            # 진입: 다음 거래일 시가
            future_all = df_daily[df_daily.index > signal_date]
            if len(future_all) < 2:
                continue
            entry_price = float(future_all["Open"].iloc[0])
            entry_date  = future_all.index[0]

            exit_window = future_all.iloc[1: HORIZON + 1]
            if exit_window.empty:
                continue

            raw_pnl, reason = _barrier_exit(exit_window, entry_price)
            net_pnl         = _apply_costs(raw_pnl, is_korean)

            exit_date  = exit_window.index[min(HORIZON - 1, len(exit_window) - 1)]
            exit_price = (
                entry_price * (1 + TP_PCT)   if reason == "tp"
                else entry_price * (1 - SL_PCT) if reason == "sl"
                else float(exit_window["Close"].iloc[-1])
            )
            in_position_until = exit_date
            trades.append({
                "ticker":      ticker,
                "fold":        fold_i,
                "signal_date": str(signal_date.date()),
                "entry_date":  str(entry_date.date()),
                "exit_date":   str(exit_date.date()),
                "entry_price": round(entry_price, 4),
                "exit_price":  round(exit_price, 4),
                "raw_pnl_pct": round(raw_pnl * 100, 3),
                "net_pnl_pct": round(net_pnl * 100, 3),
                "exit_reason": reason,
                "is_win":      int(net_pnl > 0),
                "strategy":    strategy,
                "triggers":    ",".join(triggers),
            })

    return trades


def run_f1_comparison(stocks: dict | None = None) -> str:
    """
    F1: B0(Random) / B1(Trigger) / B2(Rule MA60) / ML 4종 walk-forward 비교.

    공정 비교 보장:
      - 동일 유니버스 (정적 30종목)
      - 동일 기간 (5년 walk-forward)
      - 동일 트리거 베이스 (5종 기술적 트리거)
      - 동일 청산 규칙 (TP=7% / SL=7% / 7거래일)
      - 동일 비용 모델
      - 차이: 2차 필터만 다름 (없음 / 랜덤 / MA60 / ML)
    """
    if stocks is None:
        from signals.krx_universe import get_krx_backtest_universe
        stocks = get_krx_backtest_universe(use_static=True)
        if not stocks:
            stocks = _FALLBACK_STOCKS

    logger.info("=== F1 비교 백테스트 시작 (%d종목 × 4전략) ===", len(stocks))

    all_trades: dict[str, list[dict]] = {s: [] for s in ("random", "trigger", "rule", "ml")}

    for ticker, name in stocks.items():
        try:
            df = _fetch_daily(ticker)
        except Exception as e:
            logger.warning("[%s] 데이터 실패: %s", ticker, e)
            continue
        min_rows = (TRAIN_MONTHS + TEST_MONTHS) * 20 + HORIZON
        if len(df) < min_rows:
            continue

        logger.info("[%s] F1 시작...", ticker)
        for strat in ("random", "trigger", "rule", "ml"):
            t = _walk_strategy(ticker, df, strategy=strat)
            all_trades[strat].extend(t)
            logger.info("  [%s] %s: %d건", ticker, strat, len(t))

    return _format_f1_result(all_trades, stocks)


def _format_f1_result(all_trades: dict[str, list[dict]], stocks: dict) -> str:
    """F1 비교표 + ML 기여도 판정."""
    labels = {"random": "B0 Random", "trigger": "B1 Trigger", "rule": "B2 Rule(MA60)", "ml": "ML"}
    metrics = {s: _strategy_metrics(all_trades[s]) for s in ("random", "trigger", "rule", "ml")}

    # ML vs B1 부트스트랩 차이 CI
    ml_pnl = np.array([t["net_pnl_pct"] for t in all_trades["ml"]]) / 100 if all_trades["ml"] else np.array([0.0])
    b1_pnl = np.array([t["net_pnl_pct"] for t in all_trades["trigger"]]) / 100 if all_trades["trigger"] else np.array([0.0])
    diff_low, diff_high = _bootstrap_diff_ci(ml_pnl, b1_pnl)
    ml_adds_value = diff_low > 0

    lines = [
        "📊 <b>F1 — ML 기여도 비교 (공정 비교: 트리거 베이스 동일)</b>",
        f"유니버스: {len(stocks)}종목 | 기간: 5년 walk-forward",
        "",
        f"{'전략':<14} {'건수':>5} {'승률':>6} {'세후EV':>8} {'CI 하단':>8} {'CI 상단':>8} {'Sharpe':>7} {'MDD':>7}",
        "─" * 65,
    ]
    for s in ("random", "trigger", "rule", "ml"):
        m = metrics[s]
        lines.append(
            f"{labels[s]:<14} {m['n']:>5} {m['win_rate']:>5.1f}% {m['ev']*100:>+7.3f}% "
            f"{m['ci_low']*100:>+7.3f}% {m['ci_high']*100:>+7.3f}% "
            f"{m['sharpe']:>7.2f} {m['mdd']:>+6.1f}%"
        )

    ml_ev = metrics["ml"]["ev"] * 100
    b1_ev = metrics["trigger"]["ev"] * 100
    lines += [
        "",
        "<b>ML vs B1 기여도 분석</b>",
        f"  ML EV − B1 EV = {ml_ev - b1_ev:+.3f}%p",
        f"  차이 95% CI: [{diff_low*100:+.3f}%, {diff_high*100:+.3f}%]",
    ]

    if ml_adds_value:
        verdict = "✅ ML이 B1 대비 통계적으로 유의한 양의 기여 (CI 하단 > 0)"
    elif diff_high < 0:
        verdict = "❌ ML이 B1보다 통계적으로 열등 — ML이 오히려 손해를 입히고 있음"
    else:
        verdict = "⚠️ ML과 B1 통계적 동등 (CI가 0을 포함) — ML의 기여 불명확"

    lines += [f"  → {verdict}", ""]

    # 판정 경로
    ml_m  = metrics["ml"]
    b1_m  = metrics["trigger"]
    if ml_m["ev"] <= b1_m["ev"] or not ml_adds_value:
        path = "경로 A 또는 C 유력 — ML이 B1 대비 기여 없음. F2·F3 추가 진단 후 최종 결정."
    else:
        path = "경로 B 또는 C 유력 — ML이 B1 대비 양의 기여. F2 종목별 분해 필요."

    lines.append(f"<b>F4 예비 경로 판정:</b> {path}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Phase F2 — 종목별 ML vs B1 분해 + 삼성생명 개별 분석
# ─────────────────────────────────────────────────────────────────────────────

def run_f2_breakdown(stocks: dict | None = None) -> str:
    """
    F2: 종목별 ML vs B1 성과 분해.

    출력:
      1) 종목별 B1·ML 건수·EV·Δ 비교표 (30행)
      2) 삼성생명(032830.KS) 폴드별 상세 분석
      3) ML EV ≥ B1 EV인 종목 vs 하위 종목 구분
    """
    if stocks is None:
        from signals.krx_universe import get_krx_backtest_universe
        stocks = get_krx_backtest_universe(use_static=True)
        if not stocks:
            stocks = _FALLBACK_STOCKS

    logger.info("=== F2 종목별 분해 시작 (%d종목) ===", len(stocks))

    per_ticker: dict[str, tuple[str, dict[str, list[dict]]]] = {}

    for ticker, name in stocks.items():
        try:
            df = _fetch_daily(ticker)
        except Exception as e:
            logger.warning("[%s] 데이터 실패: %s", ticker, e)
            continue
        min_rows = (TRAIN_MONTHS + TEST_MONTHS) * 20 + HORIZON
        if len(df) < min_rows:
            continue

        tt: dict[str, list[dict]] = {}
        for strat in ("trigger", "ml"):
            tt[strat] = _walk_strategy(ticker, df, strategy=strat)
        per_ticker[ticker] = (name, tt)
        logger.info("[%s] B1=%d건 ML=%d건", ticker, len(tt["trigger"]), len(tt["ml"]))

    return _format_f2_result(per_ticker)


def _format_f2_result(per_ticker: dict) -> str:
    lines = ["📊 <b>F2 — 종목별 ML vs B1 분해</b>", ""]
    lines.append(f"{'종목명':<12} {'B1건':>5} {'B1 EV':>7} {'ML건':>5} {'ML EV':>7} {'Δ EV':>8}  {'판정'}")
    lines.append("─" * 68)

    ml_better: list[str] = []
    b1_better: list[str] = []
    rows = []

    for ticker, (name, tt) in sorted(per_ticker.items(), key=lambda x: x[1][0]):
        b1m = _strategy_metrics(tt["trigger"])
        mlm = _strategy_metrics(tt["ml"])
        delta = (mlm["ev"] - b1m["ev"]) * 100
        verdict = "✅ ML↑" if delta > 0 else "❌ ML↓"
        if delta > 0:
            ml_better.append(name)
        else:
            b1_better.append(name)
        rows.append((name, b1m["n"], b1m["ev"] * 100, mlm["n"], mlm["ev"] * 100, delta, verdict))

    for name, bn, bev, mn, mev, delta, verdict in sorted(rows, key=lambda r: r[5]):
        lines.append(
            f"{name:<12} {bn:>5} {bev:>+6.3f}% {mn:>5} {mev:>+6.3f}% {delta:>+7.3f}%p  {verdict}"
        )

    lines += [
        "",
        f"ML > B1 종목 ({len(ml_better)}개): {', '.join(ml_better) if ml_better else '없음'}",
        f"ML < B1 종목 ({len(b1_better)}개): {', '.join(b1_better) if b1_better else '없음'}",
        "",
    ]

    # 삼성생명(032830.KS) 폴드별 상세
    sl_name  = "삼성생명"
    sl_key   = "032830.KS"
    if sl_key in per_ticker:
        _, sl_tt = per_ticker[sl_key]
        lines.append(f"<b>삼성생명({sl_key}) 폴드별 상세</b>")
        lines.append(f"{'폴드':>4} {'B1건':>5} {'B1 EV':>7} {'ML건':>5} {'ML EV':>7}")
        lines.append("─" * 36)

        folds_b1: dict[int, list] = {}
        folds_ml: dict[int, list] = {}
        for t in sl_tt["trigger"]:
            folds_b1.setdefault(t["fold"], []).append(t)
        for t in sl_tt["ml"]:
            folds_ml.setdefault(t["fold"], []).append(t)

        all_folds = sorted(set(list(folds_b1.keys()) + list(folds_ml.keys())))
        for f in all_folds:
            b1f = _strategy_metrics(folds_b1.get(f, []))
            mlf = _strategy_metrics(folds_ml.get(f, []))
            lines.append(
                f"F{f:>3} {b1f['n']:>5} {b1f['ev']*100:>+6.3f}% {mlf['n']:>5} {mlf['ev']*100:>+6.3f}%"
            )

        b1_total = _strategy_metrics(sl_tt["trigger"])
        ml_total = _strategy_metrics(sl_tt["ml"])
        lines += [
            "─" * 36,
            f"합계  {b1_total['n']:>5} {b1_total['ev']*100:>+6.3f}% {ml_total['n']:>5} {ml_total['ev']*100:>+6.3f}%",
            f"  → 삼성생명 ML Δ = {(ml_total['ev'] - b1_total['ev'])*100:+.3f}%p",
        ]
    else:
        lines.append(f"⚠️ {sl_name} 데이터 없음")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Phase F3 — 피처 중요도 + 임계값 분석
# ─────────────────────────────────────────────────────────────────────────────

def _extract_xgb(model) -> object | None:
    """CalibratedClassifierCV → FrozenEstimator → XGBClassifier 추출."""
    try:
        inner = model.calibrated_classifiers_[0].estimator
        if hasattr(inner, "estimator"):
            return inner.estimator
        if hasattr(inner, "feature_importances_"):
            return inner
    except Exception:
        pass
    if hasattr(model, "feature_importances_"):
        return model
    return None


def run_f3_analysis(stocks: dict | None = None) -> str:
    """
    F3: 피처 중요도(fold 안정성) + 임계값 분석.

    Part 1 — 피처 중요도
      저장된 wf 모델 파일에서 XGBoost gain importance 추출.
      fold간 mean ± std → 안정적인 피처 vs 불안정 피처 판별.

    Part 2 — 임계값 분석
      ML 전략을 0.50/0.55/0.60/0.65 임계값으로 실행.
      건수·EV·Sharpe 변화 추적 (과적합 여부 확인).
    """
    import os, pickle
    from ml.features import FEATURE_COLS

    if stocks is None:
        from signals.krx_universe import get_krx_backtest_universe
        stocks = get_krx_backtest_universe(use_static=True)
        if not stocks:
            stocks = _FALLBACK_STOCKS

    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "models")
    lines: list[str] = ["📊 <b>F3 — 피처 중요도 + 임계값 분석</b>", ""]

    # ── Part 1: 피처 중요도 (fold 안정성) ──────────────────────────────────
    all_imp: dict[str, list[float]] = {f: [] for f in FEATURE_COLS}
    model_count = 0

    for ticker in stocks:
        slug = ticker.replace(".", "_")
        for fold_i in range(20):
            path = os.path.join(model_dir, f"{slug}_wf_f{fold_i}.pkl")
            if not os.path.exists(path):
                break
            try:
                with open(path, "rb") as fh:
                    saved = pickle.load(fh)
                xgb = _extract_xgb(saved["model"])
                if xgb is None:
                    continue
                fi = xgb.feature_importances_
                if len(fi) == len(FEATURE_COLS):
                    for k, v in zip(FEATURE_COLS, fi):
                        all_imp[k].append(float(v))
                    model_count += 1
            except Exception:
                continue

    lines += [
        f"<b>Part 1 — 피처 중요도 (fold 안정성)</b>  [모델 수: {model_count}개]",
        f"{'피처':<22} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}  {'안정성'}",
        "─" * 62,
    ]

    imp_summary = []
    for feat in FEATURE_COLS:
        vals = all_imp[feat]
        if not vals:
            continue
        arr = np.array(vals)
        cv = arr.std() / (arr.mean() + 1e-9)
        stability = "✅ 안정" if cv < 0.5 else ("⚠️ 보통" if cv < 1.0 else "❌ 불안정")
        imp_summary.append((feat, arr.mean(), arr.std(), arr.min(), arr.max(), stability))

    for feat, mean, std, mn, mx, stab in sorted(imp_summary, key=lambda x: -x[1]):
        lines.append(f"{feat:<22} {mean:>7.4f} {std:>7.4f} {mn:>7.4f} {mx:>7.4f}  {stab}")

    top3 = [x[0] for x in sorted(imp_summary, key=lambda x: -x[1])[:3]]
    bot3 = [x[0] for x in sorted(imp_summary, key=lambda x: x[1])[:3]]
    lines += [
        "",
        f"  상위 3 피처: {', '.join(top3)}",
        f"  하위 3 피처: {', '.join(bot3)}",
        "",
    ]

    # ── Part 2: 임계값 분석 ────────────────────────────────────────────────
    lines += [
        "<b>Part 2 — 임계값별 ML 성과</b>",
        f"{'임계값':>6} {'건수':>5} {'승률':>6} {'세후EV':>8} {'CI 하단':>8} {'CI 상단':>8} {'Sharpe':>7}",
        "─" * 58,
    ]

    THRESHOLDS = [0.50, 0.55, 0.60, 0.65]

    for thr in THRESHOLDS:
        all_trades: list[dict] = []
        for ticker, name in stocks.items():
            try:
                df = _fetch_daily(ticker)
            except Exception:
                continue
            min_rows = (TRAIN_MONTHS + TEST_MONTHS) * 20 + HORIZON
            if len(df) < min_rows:
                continue
            t = _walk_strategy(ticker, df, strategy="ml", threshold=thr)
            all_trades.extend(t)

        m = _strategy_metrics(all_trades)
        lines.append(
            f"{thr:>6.2f} {m['n']:>5} {m['win_rate']:>5.1f}% {m['ev']*100:>+7.3f}% "
            f"{m['ci_low']*100:>+7.3f}% {m['ci_high']*100:>+7.3f}% {m['sharpe']:>7.2f}"
        )
        logger.info("[F3] 임계값=%.2f → %d건, EV=%.3f%%", thr, m["n"], m["ev"] * 100)

    lines += [
        "",
        "  해석 기준: 임계값↑ → 건수↓·EV↑이면 ML이 실제 변별력 있음.",
        "             임계값↑ → 건수↓·EV 변화 없으면 ML 변별력 없음.",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Phase H2 — 레짐 베이스라인 비교 (R0/R1/R2/R3/ML)
# ─────────────────────────────────────────────────────────────────────────────

def _walk_regime_mode(ticker: str, df_daily: pd.DataFrame,
                      regime_mode: str, kospi_df: pd.DataFrame | None) -> list[dict]:
    """
    단일 종목 × 단일 레짐 모드 walk-forward.
    regime_mode: 'R0'|'R1'|'R2'|'R3'|'ML'
    """
    from ml.features import add_features, FEATURE_COLS
    from ml.regime_model import train_regime, predict_regime

    folds = _make_folds(df_daily)
    if not folds:
        return []

    is_korean = _is_korean_ticker(ticker)
    trades: list[dict] = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        # 레짐 모델 학습 (ML 모드만)
        regime_model = None
        if regime_mode == "ML" and kospi_df is not None:
            regime_model = train_regime(kospi_df, train_idx[-1])

        context_df = df_daily[df_daily.index <= test_idx[-1]]
        try:
            feat_df = add_features(context_df).dropna(subset=FEATURE_COLS)
        except Exception:
            continue
        test_feat = feat_df[feat_df.index.isin(test_idx)]
        if test_feat.empty:
            continue

        trig_map = _precompute_triggers(context_df)
        in_position_until: pd.Timestamp | None = None

        for signal_date in test_feat.index:
            if in_position_until is not None and signal_date <= in_position_until:
                continue

            triggers = trig_map.get(signal_date, [])
            if not triggers:
                continue

            # MA200 필터
            history = df_daily[df_daily.index <= signal_date]
            if len(history) >= 200:
                ma200 = float(history["Close"].rolling(200).mean().iloc[-1])
                if float(history["Close"].iloc[-1]) < ma200:
                    continue

            # 레짐 판정 (모드별)
            if kospi_df is not None:
                hist_k = kospi_df[kospi_df.index <= signal_date]
                if len(hist_k) >= 200:
                    kclose = hist_k["Close"].squeeze().astype(float)
                    if regime_mode == "R0":
                        pass  # 필터 없음
                    elif regime_mode == "R1":
                        ma200k = kclose.rolling(200).mean().iloc[-1]
                        if kclose.iloc[-1] < ma200k:
                            continue
                    elif regime_mode == "R2":
                        ma50k  = kclose.rolling(50).mean().iloc[-1]
                        ma200k = kclose.rolling(200).mean().iloc[-1]
                        if ma50k < ma200k:
                            continue
                    elif regime_mode == "R3":
                        if len(hist_k) >= 60 and kclose.iloc[-1] < kclose.iloc[-60]:
                            continue
                    elif regime_mode == "ML":
                        if not predict_regime(regime_model, kospi_df, signal_date):
                            continue

            future_all = df_daily[df_daily.index > signal_date]
            if len(future_all) < 2:
                continue

            entry_price = float(future_all["Open"].iloc[0])

            exit_window = future_all.iloc[1: HORIZON + 1]
            if exit_window.empty:
                continue

            raw_pnl, reason = _barrier_exit(exit_window, entry_price)
            net_pnl         = _apply_costs(raw_pnl, is_korean)

            exit_date  = exit_window.index[min(HORIZON - 1, len(exit_window) - 1)]
            in_position_until = exit_date
            trades.append({
                "ticker":      ticker,
                "fold":        fold_i,
                "signal_date": str(signal_date.date()),
                "net_pnl_pct": round(net_pnl * 100, 3),
                "exit_reason": reason,
                "is_win":      int(net_pnl > 0),
                "regime_mode": regime_mode,
            })

    return trades


def run_h2_comparison(stocks: dict | None = None) -> str:
    """H2: R0/R1/R2/R3/ML 레짐 모드 5종 비교."""
    if stocks is None:
        from signals.krx_universe import get_krx_backtest_universe
        stocks = get_krx_backtest_universe(use_static=True) or _FALLBACK_STOCKS

    try:
        kospi_df = _fetch_daily("069500.KS", years=6)
    except Exception:
        kospi_df = None

    MODES = ["R0", "R1", "R2", "R3", "ML"]
    all_trades: dict[str, list] = {m: [] for m in MODES}

    for ticker, name in stocks.items():
        try:
            df = _fetch_daily(ticker)
        except Exception:
            continue
        if len(df) < (TRAIN_MONTHS + TEST_MONTHS) * 20 + HORIZON:
            continue
        logger.info("[H2] %s", ticker)
        for mode in MODES:
            t = _walk_regime_mode(ticker, df, mode, kospi_df)
            all_trades[mode].extend(t)
            logger.info("  %s: %d건", mode, len(t))

    MODE_LABELS = {"R0": "R0 필터없음", "R1": "R1 MA200",
                   "R2": "R2 골든크로스", "R3": "R3 60일모멘텀", "ML": "ML 레짐"}

    lines = ["📊 <b>H2 — 레짐 방식 5종 비교</b>",
             f"{'전략':<14} {'건수':>5} {'승률':>6} {'세후EV':>8} {'CI하단':>8} {'CI상단':>8} {'Sharpe':>7}",
             "─" * 62]

    best_non_ml_ev = -np.inf
    best_non_ml    = None
    ml_pnl         = np.array([t["net_pnl_pct"] for t in all_trades["ML"]]) / 100 if all_trades["ML"] else np.array([0.0])

    mode_metrics = {}
    for mode in MODES:
        m = _strategy_metrics(all_trades[mode])
        mode_metrics[mode] = m
        lines.append(
            f"{MODE_LABELS[mode]:<14} {m['n']:>5} {m['win_rate']:>5.1f}% "
            f"{m['ev']*100:>+7.3f}% {m['ci_low']*100:>+7.3f}% "
            f"{m['ci_high']*100:>+7.3f}% {m['sharpe']:>7.2f}"
        )
        if mode != "ML" and m["ev"] > best_non_ml_ev:
            best_non_ml_ev = m["ev"]
            best_non_ml    = mode

    # ML vs 최고 단순 베이스라인 차이 CI
    best_pnl = np.array([t["net_pnl_pct"] for t in all_trades[best_non_ml]]) / 100 if all_trades[best_non_ml] else np.array([0.0])
    diff_low, diff_high = _bootstrap_diff_ci(ml_pnl, best_pnl)

    lines += [
        "",
        f"<b>ML vs 최고 단순 베이스라인({MODE_LABELS[best_non_ml]}) 차이 분석</b>",
        f"  ML EV − {best_non_ml} EV = {(mode_metrics['ML']['ev'] - best_non_ml_ev)*100:+.3f}%p",
        f"  차이 95% CI: [{diff_low*100:+.3f}%, {diff_high*100:+.3f}%]",
    ]

    if diff_low > 0:
        verdict = "✅ ML이 단순 규칙 대비 통계적으로 유의하게 우월"
    elif diff_high < 0:
        verdict = "❌ ML이 단순 규칙보다 열등 — ML 제거 권고"
    else:
        verdict = "⚠️ ML과 단순 규칙 통계적 동등 (CI ⊃ 0) — 단순 규칙으로 대체 검토"

    lines.append(f"  → {verdict}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Phase H3 — 연도별 / 종목별 / 슬리피지 안정성
# ─────────────────────────────────────────────────────────────────────────────

def run_h3_stability(stocks: dict | None = None) -> str:
    """H3: GATE C 결과 재사용 → 연도별·종목별·슬리피지 분해."""
    if stocks is None:
        from signals.krx_universe import get_krx_backtest_universe
        stocks = get_krx_backtest_universe(use_static=True) or _FALLBACK_STOCKS

    all_trades: list[dict] = []
    for ticker, name in stocks.items():
        try:
            df = _fetch_daily(ticker)
        except Exception:
            continue
        if len(df) < (TRAIN_MONTHS + TEST_MONTHS) * 20 + HORIZON:
            continue
        trades = walkforward_ticker(ticker, df)
        for t in trades:
            t["name"] = name
        all_trades.extend(trades)
        logger.info("[H3] %s: %d건", ticker, len(trades))

    lines = ["📊 <b>H3 — 안정성 분석</b>", ""]

    # ── Part 1: 연도별 EV ──────────────────────────────────────────────────
    lines.append("<b>Part 1 — 연도별 세후 EV</b>")
    lines.append("%-6s  %5s  %6s  %8s  %8s  %8s" % (
        "연도", "건수", "승률", "EV", "CI 하단", "CI 상단"))
    lines.append("─" * 56)

    from collections import defaultdict
    by_year: dict[int, list] = defaultdict(list)
    for t in all_trades:
        yr = int(str(t["signal_date"])[:4])
        by_year[yr].append(t)

    for yr in sorted(by_year):
        m = _strategy_metrics(by_year[yr])
        lines.append("%-6d  %5d  %5.1f%%  %+7.3f%%  %+7.3f%%  %+7.3f%%" % (
            yr, m["n"], m["win_rate"],
            m["ev"] * 100, m["ci_low"] * 100, m["ci_high"] * 100))

    # 연도별 양수 비율
    pos_yrs = sum(1 for yr in by_year if _strategy_metrics(by_year[yr])["ev"] > 0)
    lines.append("  양수 연도: %d/%d" % (pos_yrs, len(by_year)))

    # ── Part 2: 종목별 EV (n≥10만) ────────────────────────────────────────
    lines += ["", "<b>Part 2 — 종목별 세후 EV (n≥10)</b>"]
    lines.append("%-16s  %5s  %6s  %8s  %8s  %8s" % (
        "종목", "건수", "승률", "EV", "CI 하단", "CI 상단"))
    lines.append("─" * 66)

    by_ticker: dict[str, list] = defaultdict(list)
    for t in all_trades:
        by_ticker[t["ticker"]].append(t)

    ticker_rows = []
    for tick, tlist in by_ticker.items():
        m = _strategy_metrics(tlist)
        if m["n"] >= 10:
            name = stocks.get(tick, tick)
            ticker_rows.append((m["ev"], name, tick, m))

    ticker_rows.sort(reverse=True)
    pos_tickers = neg_tickers = 0
    for ev, name, tick, m in ticker_rows:
        if m["ev"] > 0:
            pos_tickers += 1
        else:
            neg_tickers += 1
        label = ("%s(%s)" % (name, tick))[:16]
        lines.append("%-16s  %5d  %5.1f%%  %+7.3f%%  %+7.3f%%  %+7.3f%%" % (
            label, m["n"], m["win_rate"],
            m["ev"] * 100, m["ci_low"] * 100, m["ci_high"] * 100))

    lines.append("  양수 종목: %d/%d (n≥10 기준)" % (pos_tickers, pos_tickers + neg_tickers))

    # ── Part 3: 슬리피지 민감도 ─────────────────────────────────────────────
    lines += ["", "<b>Part 3 — 슬리피지 민감도</b>"]
    lines.append("%-10s  %8s  %8s  %8s" % ("슬리피지", "세후 EV", "CI 하단", "CI 상단"))
    lines.append("─" * 42)

    base_pnl_arr = np.array([t["net_pnl_pct"] for t in all_trades]) / 100
    for slip in (0.0025, 0.0040, 0.0060, 0.0080, 0.0100):
        # 슬리피지 변경분 = (slip - 0.0025) × 2 (왕복) 추가 비용
        delta = (slip - 0.0025) * 2
        adj   = base_pnl_arr - delta
        lo, hi = _bootstrap_ci(adj)
        ev = float(adj.mean())
        tag = "✅" if ev > 0 and lo > 0 else ("⚠️" if ev > 0 else "❌")
        lines.append("%s slip=%.2f%%  %+7.3f%%  %+7.3f%%  %+7.3f%%" % (
            tag, slip * 100, ev * 100, lo * 100, hi * 100))

    # ── 최종 안정성 판정 ────────────────────────────────────────────────────
    lines.append("")
    slip_breakeven = None
    for slip in np.arange(0.0025, 0.0110, 0.0005):
        delta = (slip - 0.0025) * 2
        adj = base_pnl_arr - delta
        if adj.mean() <= 0:
            slip_breakeven = slip
            break

    if slip_breakeven:
        lines.append("슬리피지 손익분기: %.2f%%%% → 현재(0.25%%) 대비 %.2f%%%% 여유" % (
            slip_breakeven * 100, (slip_breakeven - 0.0025) * 100))
    else:
        lines.append("슬리피지 손익분기: 1.0% 이상 — 높은 슬리피지 내성")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Phase H4 — 레짐 모델 OOS AUC + 피처 안정성
# ─────────────────────────────────────────────────────────────────────────────

def run_h4_regime_diagnostics(stocks: dict | None = None) -> str:
    """H4: fold별 OOS AUC, 피처 중요도 안정성."""
    from ml.regime_model import train_regime, _build_regime_features, REGIME_FEATURES, REGIME_HORIZON

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return "H4 실패: scikit-learn 없음"

    if stocks is None:
        from signals.krx_universe import get_krx_backtest_universe
        stocks = get_krx_backtest_universe(use_static=True) or _FALLBACK_STOCKS

    try:
        kospi_df = _fetch_daily("069500.KS", years=6)
    except Exception as e:
        return "H4 실패: 코스피 데이터 로드 오류 — %s" % e

    df_ref = _fetch_daily(list(stocks.keys())[0])
    folds  = _make_folds(df_ref)

    auc_scores: list[float] = []
    feat_importances: list[np.ndarray] = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_end = train_idx[-1]
        model = train_regime(kospi_df, train_end)
        if model is None:
            continue

        # OOS AUC: test 기간 KOSPI200 데이터로 계산
        test_end = test_idx[-1]
        kospi_test = kospi_df[
            (kospi_df.index > train_end) & (kospi_df.index <= test_end)
        ]
        if len(kospi_test) < REGIME_HORIZON + 10:
            continue

        feat_full = _build_regime_features(kospi_df[kospi_df.index <= test_end])
        feat_test = feat_full[feat_full.index > train_end]

        close_full = kospi_df["Close"].squeeze().astype(float)
        future_ret = close_full.shift(-REGIME_HORIZON) / close_full - 1
        true_label = (future_ret >= 0).astype(int)
        true_test  = true_label[true_label.index > train_end]
        true_test  = true_test[true_test.index <= test_end]
        true_test  = true_test.iloc[:-REGIME_HORIZON]  # 미래 없는 행 제거

        common_idx = feat_test.index.intersection(true_test.index)
        if len(common_idx) < 10:
            continue

        X_oos = feat_test.loc[common_idx, REGIME_FEATURES].dropna()
        y_oos = true_test.loc[X_oos.index]
        if y_oos.nunique() < 2:
            continue

        prob = model.predict_proba(X_oos.values.astype("float32"))[:, 1]
        auc  = roc_auc_score(y_oos, prob)
        auc_scores.append(auc)
        feat_importances.append(model.feature_importances_)
        logger.info("[H4] fold %d  OOS AUC=%.3f", fold_i, auc)

    lines = ["📊 <b>H4 — 레짐 모델 내재 진단</b>", ""]

    # ── OOS AUC ─────────────────────────────────────────────────────────────
    lines.append("<b>Part 1 — OOS AUC (fold별)</b>")
    if auc_scores:
        for i, auc in enumerate(auc_scores):
            tag = "✅" if auc >= 0.55 else ("⚠️" if auc >= 0.50 else "❌")
            lines.append("  %s Fold %2d: AUC=%.3f" % (tag, i, auc))
        mean_auc = float(np.mean(auc_scores))
        std_auc  = float(np.std(auc_scores))
        lines.append("")
        lines.append("  평균 AUC: %.3f ± %.3f" % (mean_auc, std_auc))
        if mean_auc >= 0.55:
            auc_verdict = "✅ AUC≥0.55 — 레짐 예측 유의미"
        elif mean_auc >= 0.50:
            auc_verdict = "⚠️ AUC 0.50~0.55 — 약한 신호"
        else:
            auc_verdict = "❌ AUC<0.50 — 무작위 이하, 레짐 모델 폐기 검토"
        lines.append("  → " + auc_verdict)
    else:
        lines.append("  AUC 계산 실패 (데이터 부족)")

    # ── 피처 중요도 안정성 ──────────────────────────────────────────────────
    lines += ["", "<b>Part 2 — 피처 중요도 안정성 (fold간 std/mean)</b>"]
    if feat_importances:
        arr = np.array(feat_importances)
        means = arr.mean(axis=0)
        stds  = arr.std(axis=0)
        cvs   = stds / (means + 1e-8)
        order = np.argsort(-means)

        lines.append("%-16s  %6s  %6s  %5s  판정" % ("피처", "mean", "std", "CV"))
        lines.append("─" * 50)
        for idx in order:
            feat_name = REGIME_FEATURES[idx]
            tag = "✅" if cvs[idx] < 0.5 else ("⚠️" if cvs[idx] < 1.0 else "❌ 불안정")
            lines.append("%-16s  %.4f  %.4f  %.2f  %s" % (
                feat_name, means[idx], stds[idx], cvs[idx], tag))

        stable_n = sum(1 for cv in cvs if cv < 0.5)
        lines.append("")
        lines.append("  안정 피처(CV<0.5): %d/%d" % (stable_n, len(REGIME_FEATURES)))
    else:
        lines.append("  피처 중요도 데이터 없음")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# B — 익일 시초가 진입 실험 (B2·B3·B4)
# ─────────────────────────────────────────────────────────────────────────────

def _run_universe(stocks: dict, horizon: int, slip: float) -> list[dict]:
    """전역 HORIZON/SLIPPAGE_PCT를 임시 변경해 유니버스 전체 백테스트. trades 반환."""
    import backtest.backtest_walkforward as _bw
    old_h, old_s = _bw.HORIZON, _bw.SLIPPAGE_PCT
    _bw.HORIZON      = horizon
    _bw.SLIPPAGE_PCT = slip
    try:
        all_trades: list[dict] = []
        for ticker, _name in stocks.items():
            try:
                df = _fetch_daily(ticker)
            except Exception:
                continue
            min_rows = (TRAIN_MONTHS + TEST_MONTHS) * 20 + horizon
            if len(df) < min_rows:
                continue
            logger.info("[B] %s  horizon=%d slip=%.4f", ticker, horizon, slip)
            all_trades.extend(walkforward_ticker(ticker, df))
        return all_trades
    finally:
        _bw.HORIZON      = old_h
        _bw.SLIPPAGE_PCT = old_s


def _b_metrics(trades: list[dict]) -> dict:
    """거래 목록 → EV·CI·승률·MDD 딕셔너리."""
    closed = [t for t in trades if t.get("net_pnl_pct") is not None]
    if not closed:
        return {"n": 0, "ev": 0, "wr": 0, "ci_low": 0, "ci_high": 0, "mdd": 0}
    pnl = np.array([float(t["net_pnl_pct"]) / 100 for t in closed])
    n   = len(pnl)
    ev  = float(pnl.mean())
    wr  = float(np.mean([t["is_win"] for t in closed]))
    ci_low, ci_high = _bootstrap_ci(pnl)
    # MDD (누적 수익 곡선 기준)
    cum = np.cumsum(pnl)
    mdd = float(np.min(cum - np.maximum.accumulate(cum)))
    return {"n": n, "ev": ev, "wr": wr,
            "ci_low": ci_low, "ci_high": ci_high, "mdd": mdd}


def _load_stocks_for_b(stocks: dict | None) -> dict:
    if stocks:
        return stocks
    try:
        from signals.krx_universe import get_krx_backtest_universe
        s = get_krx_backtest_universe(top_n=50, use_static=True)
        if s:
            return s
    except Exception:
        pass
    try:
        from config import STOCKS
        if STOCKS:
            return dict(STOCKS)
    except Exception:
        pass
    return _FALLBACK_STOCKS


def run_b2_horizon_comparison(stocks: dict | None = None) -> str:
    """B2: HORIZON 7·14·20 × 익일 시초가(slip=0.05%) 비교."""
    stocks = _load_stocks_for_b(stocks)
    logger.info("=== B2 HORIZON 비교 시작 (%d종목) ===", len(stocks))

    rows = []
    for horizon in [7, 14, 20]:
        trades = _run_universe(stocks, horizon=horizon, slip=0.0005)
        m      = _b_metrics(trades)
        rows.append((horizon, m))
        logger.info("[B2] H=%d  n=%d  EV=%.3f%%  CI=[%.3f%%, %.3f%%]",
                    horizon, m["n"], m["ev"]*100, m["ci_low"]*100, m["ci_high"]*100)

    lines = ["📊 <b>B2 — HORIZON 비교 (익일 시초가 slip=0.05%%)</b>", ""]
    lines.append("%-8s %6s %6s %8s %8s %8s %7s" % (
        "HORIZON", "거래수", "승률", "세후EV", "CI하단", "CI상단", "MDD"))
    lines.append("─" * 60)
    for horizon, m in rows:
        ev_flag = "✅" if m["ci_low"] > 0 else ("⚠️" if m["ev"] > 0 else "❌")
        lines.append("%-8s %6d %5.1f%% %+7.3f%% %+7.3f%% %+7.3f%% %+6.3f%%  %s" % (
            "%d일" % horizon, m["n"], m["wr"]*100,
            m["ev"]*100, m["ci_low"]*100, m["ci_high"]*100,
            m["mdd"]*100, ev_flag))
    return "\n".join(lines)


def run_b3_slip_sensitivity(stocks: dict | None = None) -> str:
    """B3: HORIZON {7,14,20} × 슬리피지 {0.05,0.10,0.15,0.20}% = 12칸."""
    stocks  = _load_stocks_for_b(stocks)
    slips   = [0.0005, 0.0010, 0.0015, 0.0020]
    horizons = [7, 14, 20]
    logger.info("=== B3 슬리피지 민감도 시작 (%d종목) ===", len(stocks))

    # trades 캐시: horizon → list[dict]
    cache: dict[int, list[dict]] = {}
    for h in horizons:
        cache[h] = _run_universe(stocks, horizon=h, slip=0.0)  # 비용 없이 raw 수집

    lines = ["📊 <b>B3 — 슬리피지 민감도 (세후 EV / CI 하단)</b>", ""]
    header = "%-8s" % "HORIZON"
    for s in slips:
        header += "  slip=%.2f%%" % (s * 100)
    lines.append(header)
    lines.append("─" * 65)

    # 손익분기 슬리피지 계산 헬퍼
    def _breakeven(raw_evs: list[float], slips_: list[float]) -> str:
        for i in range(len(slips_) - 1):
            if raw_evs[i] > 0 >= raw_evs[i + 1]:
                # 선형 보간
                x0, y0 = slips_[i], raw_evs[i]
                x1, y1 = slips_[i + 1], raw_evs[i + 1]
                be = x0 - y0 * (x1 - x0) / (y1 - y0)
                return "%.2f%%" % (be * 100)
        if raw_evs[-1] > 0:
            return ">%.2f%%" % (slips_[-1] * 100)
        return "<%.2f%%" % (slips_[0] * 100)

    for h in horizons:
        raw_trades = cache[h]
        ev_row = []
        row_str = "%-8s" % ("%d일" % h)
        for s in slips:
            # 비용 후처리: raw_pnl에 cost 재적용
            import backtest.backtest_walkforward as _bw
            old_s = _bw.SLIPPAGE_PCT
            _bw.SLIPPAGE_PCT = s
            try:
                adj_trades = []
                for t in raw_trades:
                    raw = float(t["raw_pnl_pct"]) / 100
                    is_kr = _is_korean_ticker(t["ticker"])
                    net   = _apply_costs(raw, is_kr) * 100
                    adj_trades.append({**t, "net_pnl_pct": round(net, 3),
                                       "is_win": int(net > 0)})
            finally:
                _bw.SLIPPAGE_PCT = old_s
            m = _b_metrics(adj_trades)
            ev_row.append(m["ev"])
            ci_flag = "✅" if m["ci_low"] > 0 else ("⚠️" if m["ev"] > 0 else "❌")
            row_str += "  %+.3f%%(%s)" % (m["ev"]*100, ci_flag)
        lines.append(row_str)
        lines.append("  손익분기: %s" % _breakeven(ev_row, slips))

    return "\n".join(lines)


def run_b4_gate_comparison(stocks: dict | None = None) -> str:
    """B4: 기존 GATE C (slip=0.25%) vs 신규 (slip=0.05%) 직접 비교."""
    stocks = _load_stocks_for_b(stocks)
    logger.info("=== B4 GATE C 비교 시작 (%d종목) ===", len(stocks))

    # 신규: HORIZON=7, slip=0.05%
    new_trades = _run_universe(stocks, horizon=7, slip=0.0005)
    nm = _b_metrics(new_trades)

    # 기존 GATE C 결과 (H5 최종 판정값, 하드코딩)
    OLD = {"ev": 0.00667, "ci_low": 0.00030, "ci_high": 0.01280,
           "n": 327, "wr": 0.583, "breakeven_slip": 0.0060}

    lines = [
        "📊 <b>B4 — GATE C 기존 vs 신규 직접 비교</b>", "",
        "%-16s %10s %10s" % ("항목", "기존(0.25%%)", "신규(0.05%%)"),
        "─" * 42,
        "%-16s %+9.3f%% %+9.3f%%  %s" % (
            "세후 EV",
            OLD["ev"]*100, nm["ev"]*100,
            "▲" if nm["ev"] > OLD["ev"] else "▼"),
        "%-16s %+9.3f%% %+9.3f%%  %s" % (
            "CI 하단",
            OLD["ci_low"]*100, nm["ci_low"]*100,
            "✅" if nm["ci_low"] > 0 else "❌"),
        "%-16s %+9.3f%% %+9.3f%%  %s" % (
            "CI 상단",
            OLD["ci_high"]*100, nm["ci_high"]*100, ""),
        "%-16s %9d  %9d" % ("거래수", OLD["n"], nm["n"]),
        "%-16s %9.1f%% %9.1f%%" % ("승률", OLD["wr"]*100, nm["wr"]*100),
        "%-16s %9s  %9s" % (
            "손익분기슬립",
            "~%.2f%%" % (OLD["breakeven_slip"]*100), "계산중"),
        "",
    ]

    # 판정
    ev_ok  = nm["ev"] > 0.005
    ci_ok  = nm["ci_low"] > 0
    if ev_ok and ci_ok:
        verdict = "✅ 계속 진행 — 신규 전략으로 페이퍼 교체 권고"
    elif nm["ev"] > 0 and nm["ci_low"] > -0.002:
        verdict = "⚠️ 부분 진행 — 페이퍼에서 실측 슬리피지 먼저 확인"
    else:
        verdict = "❌ 중단 — 진입 타이밍 변경만으로 한계. 전략 컨셉 재검토"
    lines.append(verdict)

    return "\n".join(lines)


def run_b5_equity_curve(stocks: dict | None = None) -> str:
    """
    B5: Kelly 포지션 사이징 반영 누적 수익률 곡선 + 연도별 총 수익률.

    Kelly fraction = (p×b - q) / b  (전체 기간 통계 기준)
    Equity curve  = 각 거래를 exit_date 순서로 복리 반영
    동시 포지션   = 각 베팅을 독립으로 처리 (이론적 Kelly 자금 배분)
    """
    stocks = _load_stocks_for_b(stocks)
    logger.info("=== B5 Equity Curve 시작 (%d종목) ===", len(stocks))

    trades = _run_universe(stocks, horizon=HORIZON, slip=SLIPPAGE_PCT)
    closed = sorted(
        [t for t in trades if t.get("net_pnl_pct") is not None],
        key=lambda t: t["exit_date"],
    )

    if not closed:
        return "거래 없음"

    n        = len(closed)
    pnl_arr  = np.array([t["net_pnl_pct"] / 100 for t in closed])
    wins     = pnl_arr[pnl_arr > 0]
    losses   = pnl_arr[pnl_arr <= 0]

    # Kelly 파라미터
    p       = len(wins) / n
    avg_win = float(wins.mean())        if len(wins) > 0   else 0.0
    avg_los = float(abs(losses.mean())) if len(losses) > 0 else 1e-9
    b       = avg_win / avg_los
    kelly   = max(0.0, min(1.0, (p * b - (1 - p)) / b))

    fracs = [
        ("Full Kelly",    kelly),
        ("Half Kelly",    kelly / 2),
        ("Quarter Kelly", kelly / 4),
    ]

    # ── Equity 계산 (exit_date 순) ──────────────────────────────────────
    equity: dict[str, float] = {name: 1.0 for name, _ in fracs}
    curve_hk: list[float]    = [1.0]   # Half Kelly 곡선

    years = sorted(set(t["exit_date"][:4] for t in closed))
    yr_start: dict[str, dict[str, float]] = {y: {} for y in years}
    yr_end:   dict[str, dict[str, float]] = {y: {} for y in years}

    for t in closed:
        pnl  = t["net_pnl_pct"] / 100
        year = t["exit_date"][:4]
        for name, frac in fracs:
            if name not in yr_start[year]:
                yr_start[year][name] = equity[name]
            equity[name] *= (1 + pnl * frac)
            yr_end[year][name] = equity[name]
        curve_hk.append(equity["Half Kelly"])

    n_years = len(years)

    # ── 연도별 수익률 표 ────────────────────────────────────────────────
    lines = [
        "📊 <b>B5 — Kelly 포지션 사이징 누적 수익률</b>",
        "",
        "Kelly: p=%.1f%%  b=%.2f  → Full=%.1f%%  Half=%.1f%%  Quarter=%.1f%%" % (
            p*100, b, kelly*100, kelly/2*100, kelly/4*100),
        "",
        "%-6s  %5s  %10s  %10s  %12s" % ("연도", "건수", "Full K", "Half K", "Quarter K"),
        "─" * 50,
    ]

    for year in years:
        n_yr = sum(1 for t in closed if t["exit_date"].startswith(year))
        row  = "%-6s  %5d" % (year, n_yr)
        for name, _ in fracs:
            s = yr_start[year].get(name)
            e = yr_end[year].get(name)
            if s and e and s > 0:
                yr_ret = (e / s - 1) * 100
                flag   = "✅" if yr_ret > 0 else "❌"
                row   += "  %+8.1f%% %s" % (yr_ret, flag)
            else:
                row += "  %10s" % "N/A"
        lines.append(row)

    lines.append("─" * 50)

    # 전체 누적
    row = "%-6s  %5d" % ("전체", n)
    for name, _ in fracs:
        row += "  %+8.1f%%  " % ((equity[name] - 1) * 100)
    lines.append(row)

    # CAGR
    lines += ["", "CAGR (연환산, %d년 기준):" % n_years]
    for name, _ in fracs:
        cagr = (equity[name] ** (1 / n_years) - 1) * 100
        lines.append("  %-14s: %+.1f%%" % (name, cagr))

    # Sharpe (√(252/HORIZON) 연환산)
    ann_factor = np.sqrt(252 / HORIZON)
    lines += ["", "Sharpe (연환산):"]
    for name, frac in fracs:
        sized  = pnl_arr * frac
        sharpe = (sized.mean() / (sized.std() + 1e-9)) * ann_factor
        lines.append("  %-14s: %.2f" % (name, sharpe))

    # MDD (Half Kelly)
    hk_frac   = kelly / 2
    cum_hk    = np.cumprod(1 + pnl_arr * hk_frac)
    peak      = np.maximum.accumulate(cum_hk)
    mdd_hk    = float(np.min((cum_hk - peak) / peak)) * 100
    lines.append("\nMDD (Half Kelly): %+.1f%%" % mdd_hk)

    # ── ASCII Equity Curve (Half Kelly) ────────────────────────────────
    lines += ["", "<b>Half Kelly 누적 곡선 (거래 종료 기준)</b>"]
    H, W   = 12, 64
    step   = max(1, len(curve_hk) // W)
    sample = [curve_hk[i] for i in range(0, len(curve_hk), step)][:W]
    hi, lo = max(sample), min(sample)

    for row_i in range(H, -1, -1):
        thr   = lo + (hi - lo) * row_i / H
        label = "%+5.0f%%" % ((thr - 1) * 100) if row_i % 3 == 0 else "      "
        bar   = "".join("█" if v >= thr else " " for v in sample)
        lines.append("%s |%s|" % (label, bar))

    lines.append("      +" + "─" * len(sample) + "+")
    lines.append("      시작%s종료" % (" " * (len(sample) - 4)))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# G — 비대칭 배리어 그리드 (G1·G3·G4)
# ─────────────────────────────────────────────────────────────────────────────

def _g_metrics(closed: list[dict]) -> dict:
    """그리드 분석용 확장 지표 (RR 실측·Sharpe·MDD·연도별 EV 포함)."""
    if not closed:
        return {"n": 0}

    pnl  = np.array([float(t["net_pnl_pct"]) / 100 for t in closed])
    n    = len(pnl)
    ev   = float(pnl.mean())
    wr   = float(np.mean([t["is_win"] for t in closed]))
    ci_l, ci_h = _bootstrap_ci(pnl)

    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_w  = float(wins.mean())        if len(wins)   > 0 else 0.0
    avg_l  = float(abs(losses.mean())) if len(losses) > 0 else 1e-9
    rr     = avg_w / avg_l

    # Half Kelly Sharpe / MDD
    b       = rr
    kelly   = max(0.0, min(1.0, (wr * b - (1 - wr)) / b))
    hk      = kelly / 2
    sized   = pnl * hk
    ann     = np.sqrt(252 / HORIZON)
    sharpe  = (sized.mean() / (sized.std() + 1e-9)) * ann
    cum     = np.cumprod(1 + sized)
    peak    = np.maximum.accumulate(cum)
    mdd     = float(np.min((cum - peak) / peak)) * 100

    # 연도별 EV (exit_date 기준)
    years = sorted(set(t["exit_date"][:4] for t in closed))
    yr_ev: dict[str, float] = {}
    for yr in years:
        yp = np.array([float(t["net_pnl_pct"]) / 100
                       for t in closed if t["exit_date"].startswith(yr)])
        yr_ev[yr] = float(yp.mean()) if len(yp) > 0 else 0.0

    return {"n": n, "ev": ev, "wr": wr, "ci_low": ci_l, "ci_high": ci_h,
            "rr": rr, "sharpe": sharpe, "mdd": mdd, "kelly": kelly,
            "yr_ev": yr_ev}


def run_g1_grid(stocks: dict | None = None) -> str:
    """
    G1: TP/SL 비대칭 배리어 그리드 백테스트.

    TP_RANGE = [0.07, 0.09, 0.10, 0.12, 0.15]
    SL_RANGE = [0.04, 0.05, 0.06, 0.07]
    GRID     = tp > sl 조건, 총 16개 + 현재 기준(7/7) 포함
    """
    import backtest.backtest_walkforward as _bw

    TP_RANGE = [0.07, 0.09, 0.10, 0.12, 0.15]
    SL_RANGE = [0.04, 0.05, 0.06, 0.07]
    GRID = [(tp, sl) for tp in TP_RANGE for sl in SL_RANGE if tp > sl]
    # (7,7)이 이미 포함돼 있는지 확인, 없으면 추가
    if (0.07, 0.07) not in GRID:
        GRID = [(0.07, 0.07)] + GRID

    stocks = _load_stocks_for_b(stocks)
    logger.info("=== G1 그리드 시작: %d조합 × %d종목 ===", len(GRID), len(stocks))

    rows: list[tuple] = []
    for tp, sl in GRID:
        old_tp, old_sl = _bw.TP_PCT, _bw.SL_PCT
        _bw.TP_PCT = tp
        _bw.SL_PCT = sl
        try:
            trades = _run_universe(stocks, horizon=HORIZON, slip=SLIPPAGE_PCT)
        finally:
            _bw.TP_PCT = old_tp
            _bw.SL_PCT = old_sl

        closed = [t for t in trades if t.get("net_pnl_pct") is not None]
        m = _g_metrics(closed)
        logger.info("[G1] TP=%.0f%% SL=%.0f%%  n=%d  EV=%.3f%%  CI_low=%.3f%%",
                    tp*100, sl*100, m["n"],
                    m["ev"]*100 if m["n"] else 0,
                    m["ci_low"]*100 if m["n"] else 0)
        rows.append((tp, sl, m))

    # ── 결과 표 ──────────────────────────────────────────────────────────
    lines = [
        "📊 <b>G1 — 비대칭 배리어 그리드 결과</b>",
        "",
        "%-6s %-6s %6s %6s %6s %8s %8s %6s %8s" % (
            "TP", "SL", "건수", "승률", "RR", "세후EV", "CI하단", "Sharpe", "MDD(HK)"),
        "─" * 68,
    ]

    for tp, sl, m in rows:
        if m["n"] == 0:
            lines.append("%-6s %-6s  데이터 없음" % ("%.0f%%" % (tp*100), "%.0f%%" % (sl*100)))
            continue
        is_base = (abs(tp - 0.07) < 1e-9 and abs(sl - 0.07) < 1e-9)
        # 1차 필터 마킹
        pass1 = (m["ev"] > 0.008 and m["ci_low"] > 0
                 and m["n"] >= 200 and m["mdd"] > -20)
        flag  = " ←기준" if is_base else (" ★" if pass1 else "")
        lines.append("%-6s %-6s %6d %5.1f%% %5.2f %+7.3f%% %+7.3f%% %5.2f %+7.1f%%%s" % (
            "%.0f%%" % (tp*100), "%.0f%%" % (sl*100),
            m["n"], m["wr"]*100, m["rr"],
            m["ev"]*100, m["ci_low"]*100,
            m["sharpe"], m["mdd"], flag))

    # ── G3 1차 필터 ───────────────────────────────────────────────────────
    lines += ["", "<b>G3 — 1차 필터 (EV>0.80%, CI>0, n≥200, MDD>-20%)</b>"]
    pass1_rows = [(tp, sl, m) for tp, sl, m in rows
                  if m["n"] >= 200 and m["ev"] > 0.008
                  and m["ci_low"] > 0 and m["mdd"] > -20]

    if not pass1_rows:
        lines.append("  ❌ 1차 필터 통과 조합 없음 — 현재 (7%/7%) 유지 권고")
        return "\n".join(lines)

    lines.append("  통과: %d개 조합" % len(pass1_rows))
    for tp, sl, m in pass1_rows:
        lines.append("  TP=%.0f%% SL=%.0f%%  EV=%+.3f%%  CI=%+.3f%%  n=%d  MDD=%.1f%%" % (
            tp*100, sl*100, m["ev"]*100, m["ci_low"]*100, m["n"], m["mdd"]))

    # ── G3 2차 필터: 연도별 안정성 ──────────────────────────────────────
    lines += ["", "<b>G3 — 2차 필터 (연도별 EV 안정성)</b>"]
    all_years = sorted({yr for _, _, m in pass1_rows for yr in m["yr_ev"]})
    hdr = "  %-12s" % "TP/SL"
    for yr in all_years:
        hdr += "  %6s" % yr
    hdr += "  양수연도"
    lines.append(hdr)

    pass2_rows = []
    for tp, sl, m in pass1_rows:
        row = "  %-12s" % ("%.0f%%/%.0f%%" % (tp*100, sl*100))
        pos_yrs = 0
        for yr in all_years:
            yev = m["yr_ev"].get(yr, 0) * 100
            row += "  %+5.1f%%" % yev
            if yev > 0:
                pos_yrs += 1
        row += "  %d/%d" % (pos_yrs, len(all_years))
        lines.append(row)
        if pos_yrs >= len(all_years) - 1:   # 최소 3/4 연도 양수
            pass2_rows.append((tp, sl, m))

    if not pass2_rows:
        lines.append("  ❌ 2차 필터 통과 조합 없음 — 현재 (7%/7%) 유지 권고")
        return "\n".join(lines)

    # ── G3 3차 선택: Sharpe 최대 + MDD 최소 + n 최대 ────────────────────
    lines += ["", "<b>G3 — 3차 선택 (Sharpe↑  MDD↓  거래수↑)</b>"]
    best = max(pass2_rows, key=lambda x: (x[2]["sharpe"], -x[2]["mdd"], x[2]["n"]))
    tp_b, sl_b, m_b = best
    lines.append("  ✅ 채택 후보: TP=%.0f%%  SL=%.0f%%  Sharpe=%.2f  MDD=%.1f%%  n=%d" % (
        tp_b*100, sl_b*100, m_b["sharpe"], m_b["mdd"], m_b["n"]))

    # ── G4 현재 기준 vs 후보 비교 ────────────────────────────────────────
    base = next((m for tp, sl, m in rows
                 if abs(tp - 0.07) < 1e-9 and abs(sl - 0.07) < 1e-9), None)
    if base and base["n"] > 0:
        lines += [
            "",
            "<b>G4 — 현재 (7%%/7%%) vs 후보 (%.0f%%/%.0f%%)</b>" % (tp_b*100, sl_b*100),
            "",
            "%-18s %12s %12s %6s" % ("항목", "현재(7%/7%)", "후보", "변화"),
            "─" * 52,
        ]

        def _cmp(label, v_base, v_new, higher_better=True, fmt="%+.3f%%"):
            improved = v_new > v_base if higher_better else v_new < v_base
            arrow = "▲" if improved else ("▼" if v_new != v_base else "─")
            lines.append("%-18s %12s %12s %6s" % (
                label, fmt % v_base, fmt % v_new, arrow))

        _cmp("RR(실측)",     base["rr"],       m_b["rr"],       True,  "%.2f")
        _cmp("세후 EV",      base["ev"]*100,   m_b["ev"]*100,   True)
        _cmp("CI 하단",      base["ci_low"]*100, m_b["ci_low"]*100, True)
        _cmp("승률",         base["wr"]*100,   m_b["wr"]*100,   True,  "%.1f%%")
        _cmp("거래수",       float(base["n"]), float(m_b["n"]), True,  "%.0f")
        _cmp("Sharpe",      base["sharpe"],   m_b["sharpe"],   True,  "%.2f")
        _cmp("MDD(Half K)", base["mdd"],      m_b["mdd"],      False, "%.1f%%")

        # 판정
        improve_cnt = sum([
            m_b["ev"]      > base["ev"],
            m_b["mdd"]     > base["mdd"],
            m_b["ci_low"]  > base["ci_low"],
            m_b["sharpe"]  > base["sharpe"],
        ])
        lines.append("")
        if improve_cnt >= 2:
            lines.append("✅ <b>판정: 채택</b> — 후보가 %d개 항목 개선" % improve_cnt)
            lines.append("   TP=%.0f%%  SL=%.0f%%로 전략 교체 권고" % (tp_b*100, sl_b*100))
        else:
            lines.append("⚠️ <b>판정: 현재(7%%/7%%) 유지</b> — 개선 항목 %d개 (기준 2개 미만)" % improve_cnt)

    return "\n".join(lines)


if __name__ == "__main__":
    print(run_walkforward())
