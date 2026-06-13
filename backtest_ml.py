"""
backtest_ml.py - 분봉 기반 45일 ML 전략 백테스트

흐름:
  1. 5분봉 60일치 다운로드 (yfinance)
  2. 45일 이전 일봉 5년치로 XGBoost 모델 학습
  3. 45일 구간: 분봉마다 오늘 일봉 바 합성 → 트리거 감지 → ML 예측
  4. 신호 발생 시 다음 봉 시가에 가상 매수 (하루 1건)
  5. 7거래일 후 일봉 종가로 청산
  6. 결과 요약 (승률, 손익비, 평균 수익률)

실행:
  python backtest_ml.py
  또는 텔레그램 /backtest 명령
"""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

HORIZON       = 7      # 청산 기간 (거래일)
WIN_THRESHOLD = 0.03   # 성공 기준 (3%)
MIN_WIN_PROB  = 0.60   # 봇과 동일 기준
MIN_RR        = 1.5
MIN_AUC       = 0.58   # 봇과 동일 기준
TEST_DAYS     = 45     # 백테스트 테스트 구간 (일)


# ─────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────

def _fetch_daily(ticker: str, years: int = 5) -> pd.DataFrame:
    """일봉 데이터 (ML 학습용)."""
    end   = date.today()
    start = end - relativedelta(years=years)
    df    = yf.download(ticker, start=str(start), end=str(end),
                        auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def _fetch_intraday(ticker: str) -> pd.DataFrame:
    """5분봉 60일치 (yfinance 최대 제공 범위)."""
    df = yf.download(ticker, period="60d", interval="5m",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# ─────────────────────────────────────────────
# 분봉 기반 백테스트 엔진
# ─────────────────────────────────────────────

def backtest_ticker(ticker: str, name: str) -> dict | None:
    """단일 종목 분봉 기반 45일 백테스트."""
    logger.info("  [%s] 데이터 수집 중...", ticker)

    try:
        df_min = _fetch_intraday(ticker)
    except Exception as e:
        logger.warning("  [%s] 분봉 오류: %s", ticker, e)
        return None

    if df_min.empty or len(df_min) < 50:
        logger.warning("  [%s] 분봉 데이터 부족", ticker)
        return None

    try:
        df_daily = _fetch_daily(ticker)
    except Exception as e:
        logger.warning("  [%s] 일봉 오류: %s", ticker, e)
        return None

    if len(df_daily) < 120:
        return None

    # cutoff: 45일 전 → 이전 일봉으로 ML 학습
    cutoff   = date.today() - timedelta(days=TEST_DAYS)
    train_df = df_daily[df_daily.index.date < cutoff]

    if len(train_df) < 60:
        logger.warning("  [%s] train 분량 부족", ticker)
        return None

    # ── 트리거 사전 탐지 (학습 전 빠른 필터) ────────────────────
    test_min   = df_min[df_min.index.date >= cutoff]
    test_dates = sorted(set(test_min.index.date))

    from signals.scanner import detect_triggers

    has_trigger = False
    for test_date in test_dates:
        hist = df_daily[df_daily.index.date < test_date]
        if len(hist) < 61:
            continue
        if detect_triggers(hist):
            has_trigger = True
            break

    if not has_trigger:
        return {"ticker": ticker, "name": name, "trades": 0}

    # ── 트리거 있는 종목만 모델 학습 ────────────────────────────
    logger.info("  [%s] 모델 학습 중 (train: %d행)...", ticker, len(train_df))
    try:
        from ml.model import train
        _, metrics = train(train_df, f"{ticker}_bt")
    except Exception as e:
        logger.warning("  [%s] 모델 학습 실패: %s", ticker, e)
        return None

    model_path = f"/Users/gyuyeong/quant_trader/ml/models/{ticker.replace('.','_')}_bt.pkl"
    if not os.path.exists(model_path):
        return {"ticker": ticker, "name": name, "trades": 0}

    with open(model_path, "rb") as f:
        model = pickle.load(f)["model"]

    model_auc = metrics.get("auc", 0.0)
    if model_auc < MIN_AUC:
        logger.info("  [%s] AUC %.3f < %.2f — 스킵", ticker, model_auc, MIN_AUC)
        return {"ticker": ticker, "name": name, "trades": 0}

    avg_win  = metrics["avg_win"]
    avg_loss = metrics["avg_loss"]

    from ml.features import add_features, FEATURE_COLS

    trades = []
    in_position_until = None  # 청산일까지 재진입 차단

    for test_date in test_dates:
        # 보유 중인 포지션 청산 전 재진입 금지
        if in_position_until and test_date <= in_position_until:
            continue

        day_bars = test_min[test_min.index.date == test_date]
        if len(day_bars) < 5:
            continue

        hist = df_daily[df_daily.index.date < test_date]
        if len(hist) < 61:
            continue

        signal_found = False

        for i in range(10, len(day_bars)):  # 개장 후 50분 이후부터 체크
            # 누적 분봉으로 오늘 일봉 바 합성
            accum = day_bars.iloc[:i + 1]
            today_bar = pd.DataFrame(
                [[float(accum["Open"].iloc[0]),
                  float(accum["High"].max()),
                  float(accum["Low"].min()),
                  float(accum["Close"].iloc[-1]),
                  float(accum["Volume"].sum())]],
                columns=["Open", "High", "Low", "Close", "Volume"],
                index=[pd.Timestamp(test_date)],
            )
            window = pd.concat([hist, today_bar])

            triggers = detect_triggers(window)
            if not triggers:
                continue

            try:
                w_feat = add_features(window).dropna(subset=FEATURE_COLS)
                if w_feat.empty:
                    continue
                X  = w_feat[FEATURE_COLS].iloc[[-1]].values.astype(np.float32)
                wp = float(model.predict_proba(X)[0, 1])
            except Exception:
                continue

            # 기대값 기반 손익비 (봇과 동일)
            expected_win  = avg_win  * wp
            expected_loss = avg_loss * (1 - wp)
            rr = expected_win / expected_loss if expected_loss > 0 else 0.0

            if wp < MIN_WIN_PROB or rr < MIN_RR:
                continue

            # 다음 봉 시가에 매수
            if i + 1 >= len(day_bars):
                break

            entry_price = float(day_bars["Open"].iloc[i + 1])
            future      = df_daily[df_daily.index.date > test_date]
            if len(future) < HORIZON:
                break

            # 봇과 동일한 청산 로직: 손절(-7%) / 익절(avg_win) / 기간청산(7거래일)
            exit_price  = None
            exit_date   = None
            stop_loss   = entry_price * 0.93
            take_profit = entry_price * (1 + avg_win)

            for d in range(min(HORIZON, len(future))):
                day_close = float(future["Close"].iloc[d])
                day_date  = future.index[d].date()
                if day_close <= stop_loss or day_close >= take_profit:
                    exit_price = day_close
                    exit_date  = day_date
                    break

            if exit_price is None:
                exit_price = float(future["Close"].iloc[HORIZON - 1])
                exit_date  = future.index[HORIZON - 1].date()

            pnl_pct = (exit_price - entry_price) / entry_price * 100

            in_position_until = exit_date  # 청산일까지 재진입 차단
            trades.append({
                "ticker":      ticker,
                "name":        name,
                "entry_date":  str(test_date),
                "exit_date":   str(exit_date),
                "entry_price": round(entry_price, 2),
                "exit_price":  round(exit_price, 2),
                "pnl_pct":     round(pnl_pct, 2),
                "win":         int(pnl_pct >= WIN_THRESHOLD * 100),
                "triggers":    ", ".join(triggers),
                "win_prob":    round(wp, 3),
            })
            signal_found = True
            break  # 하루 1건

        if signal_found:
            logger.info("  [%s] 신호: %s +%d건", ticker, test_date, len(trades))

    if not trades:
        return {"ticker": ticker, "name": name, "trades": 0}

    wins     = [t for t in trades if t["win"]]
    losses   = [t for t in trades if not t["win"]]
    win_rate = len(wins) / len(trades) * 100
    avg_pnl  = np.mean([t["pnl_pct"] for t in trades])
    avg_w    = np.mean([t["pnl_pct"] for t in wins])   if wins   else 0
    avg_l    = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    rr_real  = abs(avg_w / avg_l) if avg_l != 0 else 0

    return {
        "ticker":   ticker,
        "name":     name,
        "trades":   len(trades),
        "win_rate": round(win_rate, 1),
        "avg_pnl":  round(avg_pnl, 2),
        "avg_win":  round(avg_w, 2),
        "avg_loss": round(avg_l, 2),
        "rr":       round(rr_real, 2),
        "detail":   trades,
    }


# ─────────────────────────────────────────────
# 전체 종목 백테스트 + 결과 포맷
# ─────────────────────────────────────────────

def run_backtest(stocks: dict | None = None) -> str:
    """45일 분봉 ML 백테스트. 반환: 텔레그램 전송용 결과 문자열."""
    if stocks is None:
        try:
            from signals.krx_universe import get_krx_candidates
            stocks = get_krx_candidates(top_n=100)
        except Exception:
            stocks = {}
        if not stocks:
            from config import STOCKS
            stocks = STOCKS

        try:
            from signals.us_universe import get_us_candidates
            us_stocks = get_us_candidates(top_n=50)
        except Exception:
            us_stocks = {}
        if not us_stocks:
            from config import US_STOCKS
            us_stocks = US_STOCKS
        if us_stocks:
            stocks = {**stocks, **us_stocks}

    from signals.scanner import BLACKLIST
    start_date = date.today() - timedelta(days=TEST_DAYS)
    logger.info("=== 45일 분봉 ML 백테스트 시작 (%d종목) ===", len(stocks))

    results = []
    for ticker, name in stocks.items():
        if ticker in BLACKLIST:
            logger.info("  [%s] 블랙리스트 — 스킵", ticker)
            continue
        r = backtest_ticker(ticker, name)
        if r:
            results.append(r)

    if not results:
        return "⚠️ 백테스트 결과 없음 — 데이터 또는 신호 부족"

    active = [r for r in results if r.get("trades", 0) > 0]
    no_sig = [r for r in results if r.get("trades", 0) == 0]

    lines = [
        "📊 <b>45일 분봉 ML 백테스트 결과</b>",
        f"기간: {start_date} ~ {date.today()}",
        f"대상: {len(stocks)}종목 / 신호 발생: {len(active)}종목\n",
    ]

    for r in sorted(active, key=lambda x: x.get("win_rate", 0), reverse=True):
        lines.append(
            f"<b>{r['name']}</b> ({r['ticker']})\n"
            f"  거래 {r['trades']}건 | 승률 {r['win_rate']:.1f}% | "
            f"손익비 {r['rr']:.2f} | 평균 {r['avg_pnl']:+.2f}%"
        )

    if no_sig:
        lines.append(f"\n신호 없음: {', '.join(r['ticker'] for r in no_sig)}")

    if active:
        all_trades  = sum(r["trades"] for r in active)
        all_win     = sum(r["trades"] * r["win_rate"] / 100 for r in active)
        total_wr    = all_win / all_trades * 100 if all_trades > 0 else 0
        all_avg_pnl = np.mean([r["avg_pnl"] for r in active])
        lines.append(
            f"\n<b>종합</b>: 총 {all_trades}건 | "
            f"승률 {total_wr:.1f}% | 평균 {all_avg_pnl:+.2f}%"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    from signals.krx_universe import get_krx_backtest_universe
    from signals.us_universe import get_us_backtest_universe
    krx = get_krx_backtest_universe(top_n=200)
    us  = get_us_backtest_universe(top_n=100)
    stocks = {**krx, **us}
    print(run_backtest(stocks))
