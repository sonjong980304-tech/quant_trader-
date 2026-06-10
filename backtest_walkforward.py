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
  - TP=+7%, SL=-7%, 최대 7거래일
  - 당일 High/Low로 판정 (진입 다음날부터)

게이트:
  - 비용 차감 후 기대값(expectancy) ≤ 0 → 실거래 재개 불가
  - 승률이 높더라도 비용 후 음수면 명확히 보고

실행:
  python backtest_walkforward.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from datetime import date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf

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
SLIPPAGE_PCT   = 0.0025    # 진입 슬리피지 0.25% (단방향; 봇 실거래 관찰값)
STT_PCT        = 0.0018    # 증권거래세 0.18% (한국 종목 매도 시)

# Walk-forward 파라미터
TRAIN_MONTHS   = 24        # 학습 창 (개월)
TEST_MONTHS    = 3         # 테스트 창 (개월)
STEP_MONTHS    = 3         # 슬라이딩 스텝 (개월)

# 신호 필터
MIN_WIN_PROB   = 0.60
MIN_RR         = 1.5
MIN_AUC        = 0.58

# Triple-Barrier
TP_PCT         = 0.07
SL_PCT         = 0.07
HORIZON        = 7         # 최대 보유 거래일
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
    # ATR 기반 SL: 진입가 - 2×ATR, 단 고정 SL보다 넓어지지 않도록 캡
    if atr > 0:
        sl_price = max(entry_price - ATR_MULT * atr,
                       entry_price * (1 - SL_PCT))
    else:
        sl_price = entry_price * (1 - SL_PCT)

    actual_sl_pct = (entry_price - sl_price) / entry_price
    window = future_df.iloc[:HORIZON]

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
) -> list[dict]:
    """단일 종목의 모든 fold에서 매매를 시뮬레이션. trades 목록 반환."""
    from ml.model import train as ml_train
    from ml.features import add_features, FEATURE_COLS

    folds = _make_folds(df_daily)
    if not folds:
        return []

    is_korean = _is_korean_ticker(ticker)
    trades: list[dict] = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_df = df_daily.loc[train_idx]
        test_df  = df_daily.loc[test_idx]

        # ── 학습 ──────────────────────────────────────────────────
        try:
            _, metrics = ml_train(train_df, f"{ticker}_wf_f{fold_i}")
        except Exception as e:
            logger.debug("[%s] fold%d 학습 실패: %s", ticker, fold_i, e)
            continue

        if metrics.get("auc", 0) < MIN_AUC:
            continue

        avg_win  = metrics["avg_win"]
        avg_loss = metrics["avg_loss"]

        # ── 피처 계산 (테스트 구간) ────────────────────────────────
        # 컨텍스트 보존을 위해 train+test 전체에 피처 계산 후 test 구간만 사용
        context_df = df_daily[df_daily.index <= test_idx[-1]]
        try:
            feat_df = add_features(context_df).dropna(subset=FEATURE_COLS)
        except Exception:
            continue

        test_feat = feat_df[feat_df.index.isin(test_idx)]
        if test_feat.empty:
            continue

        # ── 모델 로드 (방금 학습한 모델) ──────────────────────────
        import os, pickle
        model_path = os.path.join(
            os.path.dirname(__file__),
            "ml", "models",
            f"{ticker.replace('.','_')}_wf_f{fold_i}.pkl",
        )
        if not os.path.exists(model_path):
            continue
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)["model"]

        # ── 신호 탐지 → 매매 시뮬레이션 ──────────────────────────
        in_position_until: pd.Timestamp | None = None

        for signal_date in test_feat.index:
            if in_position_until is not None and signal_date <= in_position_until:
                continue

            # MA200 추세 필터: 상승 추세가 아니면 매수 신호 차단
            history = df_daily[df_daily.index <= signal_date]
            if len(history) >= 200:
                ma200 = float(history["Close"].rolling(200).mean().iloc[-1])
                if float(history["Close"].iloc[-1]) < ma200:
                    continue

            X  = test_feat.loc[[signal_date], FEATURE_COLS].values.astype("float32")
            wp = float(model.predict_proba(X)[0, 1])

            expected_win  = avg_win  * wp
            expected_loss = avg_loss * (1 - wp)
            rr = expected_win / expected_loss if expected_loss > 0 else 0.0

            if wp < MIN_WIN_PROB or rr < MIN_RR:
                continue

            # 진입: 다음 거래일 시가
            future_all = df_daily[df_daily.index > signal_date]
            if len(future_all) < 2:
                continue

            entry_price = float(future_all["Open"].iloc[0])
            entry_date  = future_all.index[0]

            # ATR: 신호일 기준 14일 ATR
            atr_val = 0.0
            if "atr_14" in feat_df.columns and signal_date in feat_df.index:
                atr_val = float(feat_df.loc[signal_date, "atr_14"])
                if np.isnan(atr_val):
                    atr_val = 0.0

            # 청산: 진입 다음 날부터 HORIZON 거래일
            exit_window = future_all.iloc[1 : HORIZON + 1]
            if exit_window.empty:
                continue

            raw_pnl, reason = _barrier_exit(exit_window, entry_price, atr=atr_val)
            net_pnl         = _apply_costs(raw_pnl, is_korean)

            actual_sl_pct = (
                min(ATR_MULT * atr_val / entry_price, SL_PCT)
                if atr_val > 0 else SL_PCT
            )
            exit_date  = exit_window.index[min(HORIZON - 1, len(exit_window) - 1)]
            exit_price = (
                entry_price * (1 + TP_PCT)       if reason == "tp"
                else entry_price * (1 - actual_sl_pct) if reason == "sl"
                else float(exit_window["Close"].iloc[-1])
            )

            in_position_until = exit_date
            trades.append({
                "ticker":       ticker,
                "fold":         fold_i,
                "signal_date":  str(signal_date.date()),
                "entry_date":   str(entry_date.date()),
                "exit_date":    str(exit_date.date()),
                "entry_price":  round(entry_price, 4),
                "exit_price":   round(exit_price, 4),
                "raw_pnl_pct":  round(raw_pnl * 100, 3),
                "net_pnl_pct":  round(net_pnl * 100, 3),
                "exit_reason":  reason,
                "win_prob":     round(wp, 3),
                "is_win":       int(net_pnl > 0),
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
            stocks = get_krx_backtest_universe(top_n=50)
        except Exception:
            stocks = {}
        if not stocks:
            from config import STOCKS
            stocks = dict(STOCKS) if STOCKS else {}
        if not stocks:
            stocks = _FALLBACK_STOCKS
            logger.info("KRX universe/config 비어 있음 — fallback 15종목 사용")

    logger.info("=== Walk-forward 백테스트 시작 (%d종목) ===", len(stocks))

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
        trades = walkforward_ticker(ticker, df)
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
    expectancy     = avg_net / 100  # 소수점 기대값

    # 종목별 집계
    by_ticker = (
        df.groupby("ticker")
        .agg(
            trades=("is_win", "count"),
            win_rate=("is_win", lambda x: x.mean() * 100),
            avg_net=("net_pnl_pct", "mean"),
        )
        .sort_values("avg_net", ascending=False)
    )

    # 청산 유형 분포
    reason_counts = df["exit_reason"].value_counts().to_dict()

    # ── 게이트 판정 ──────────────────────────────────────────────
    gate_pass  = expectancy > 0
    gate_label = "✅ 게이트 통과 — 실거래 재개 가능" if gate_pass else \
                 "❌ 게이트 미통과 — 비용 차감 후 엣지 없음. 실거래 재개 불가"

    lines = [
        "📊 <b>Walk-forward 백테스트 결과 (비용 반영)</b>",
        f"대상: {len(stocks)}종목",
        f"총 거래: {total}건 | 승률: {win_rate:.1f}%",
        f"평균 수익(세전): {avg_raw:+.3f}%",
        f"평균 비용: -{total_cost_pct:.3f}%",
        f"평균 수익(세후): {avg_net:+.3f}%",
        f"기대값(세후): {expectancy:+.4f}",
        "",
        f"청산 유형: TP={reason_counts.get('tp',0)} / SL={reason_counts.get('sl',0)} / 기간={reason_counts.get('vertical',0)}",
        "",
        "<b>종목별 성과 (세후 평균 기준)</b>",
    ]

    for ticker, row in by_ticker.iterrows():
        name = stocks.get(ticker, ticker)
        lines.append(
            f"  {name}({ticker}): {int(row['trades'])}건 | "
            f"승률{row['win_rate']:.0f}% | 세후{row['avg_net']:+.2f}%"
        )

    lines += ["", gate_label]

    return "\n".join(lines)


if __name__ == "__main__":
    print(run_walkforward())
