"""
backtest_ml.py - 최근 1개월 ML 전략 백테스트

흐름:
  1. 관심종목 데이터 다운로드 (1년치 → 마지막 1달 테스트셋으로 사용)
  2. 1달 이전 데이터로 XGBoost 모델 학습
  3. 마지막 1달: 매일 트리거 감지 + ML 예측 → 신호 발생 시 가상 매수
  4. 7일 후 청산, 손익 기록
  5. 결과 요약 (승률, 손익비, 누적 수익률)

실행:
  python backtest_ml.py
  또는 텔레그램 /backtest 명령
"""

import logging
import warnings
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

HORIZON       = 7      # 청산 기간 (일)
WIN_THRESHOLD = 0.03   # 성공 기준 (3%)
MIN_WIN_PROB  = 0.55
MIN_RR        = 1.5


# ─────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────

def _fetch(ticker: str, years: int = 2) -> pd.DataFrame:
    end   = date.today()
    start = end - relativedelta(years=years)
    df    = yf.download(ticker, start=str(start), end=str(end),
                        auto_adjust=True, progress=False)
    df    = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


# ─────────────────────────────────────────────
# 백테스트 엔진
# ─────────────────────────────────────────────

def backtest_ticker(ticker: str, name: str) -> dict | None:
    """단일 종목 1개월 백테스트. 결과 dict 반환."""
    logger.info("  [%s] 데이터 수집 중...", ticker)
    try:
        df = _fetch(ticker, years=2)
    except Exception as e:
        logger.warning("  [%s] 데이터 오류: %s", ticker, e)
        return None

    if len(df) < 120:
        logger.warning("  [%s] 데이터 부족: %d행", ticker, len(df))
        return None

    # train / test 분리 (마지막 1개월 = test)
    cutoff    = date.today() - relativedelta(months=1)
    train_df  = df[df.index.date < cutoff]
    test_df   = df[df.index.date >= cutoff]

    if len(train_df) < 60 or len(test_df) < 5:
        logger.warning("  [%s] train/test 분량 부족", ticker)
        return None

    # 모델 학습 (train 구간)
    logger.info("  [%s] 모델 학습 중 (train: %d행)...", ticker, len(train_df))
    try:
        from ml.model import train, predict
        _, metrics = train(train_df, f"{ticker}_bt")
    except Exception as e:
        logger.warning("  [%s] 모델 학습 실패: %s", ticker, e)
        return None

    # 트리거 + ML 예측 (test 구간 날짜별)
    from signals.scanner import detect_triggers

    trades     = []
    test_dates = [d for d in test_df.index if d.date() >= cutoff]

    for i, dt in enumerate(test_dates):
        # 현재 시점까지의 데이터 (look-ahead 방지)
        window = df[df.index <= dt]
        if len(window) < 61:
            continue

        triggers = detect_triggers(window)
        if not triggers:
            continue

        # ML 예측
        from ml.features import add_features, FEATURE_COLS
        w_feat = add_features(window)
        w_feat = w_feat.dropna(subset=FEATURE_COLS)
        if w_feat.empty:
            continue

        try:
            import pickle, os
            path = f"/Users/gyuyeong/quant_trader/ml/models/{ticker.replace('.','_')}_bt.pkl"
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                data = pickle.load(f)
            model  = data["model"]
            X      = w_feat[FEATURE_COLS].iloc[[-1]].values.astype(np.float32)
            wp     = float(model.predict_proba(X)[0, 1])
        except Exception:
            continue

        avg_win  = metrics["avg_win"]
        avg_loss = metrics["avg_loss"]
        rr       = avg_win / avg_loss if avg_loss > 0 else 0

        if wp < MIN_WIN_PROB or rr < MIN_RR:
            continue

        # 가상 매수
        entry_price = float(window["Close"].iloc[-1])
        entry_date  = dt.date()

        # 7일 후 청산
        future = df[df.index.date > entry_date]
        if len(future) < HORIZON:
            continue

        exit_price = float(future["Close"].iloc[HORIZON - 1])
        exit_date  = future.index[HORIZON - 1].date()
        pnl_pct    = (exit_price - entry_price) / entry_price * 100
        win        = pnl_pct >= WIN_THRESHOLD * 100

        trades.append({
            "ticker":      ticker,
            "name":        name,
            "entry_date":  str(entry_date),
            "exit_date":   str(exit_date),
            "entry_price": round(entry_price, 2),
            "exit_price":  round(exit_price, 2),
            "pnl_pct":     round(pnl_pct, 2),
            "win":         int(win),
            "triggers":    ", ".join(triggers),
            "win_prob":    round(wp, 3),
        })

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
        "ticker":    ticker,
        "name":      name,
        "trades":    len(trades),
        "win_rate":  round(win_rate, 1),
        "avg_pnl":   round(avg_pnl, 2),
        "avg_win":   round(avg_w, 2),
        "avg_loss":  round(avg_l, 2),
        "rr":        round(rr_real, 2),
        "detail":    trades,
    }


# ─────────────────────────────────────────────
# 전체 종목 백테스트 + 결과 포맷
# ─────────────────────────────────────────────

def run_backtest(stocks: dict | None = None) -> str:
    """
    관심종목 전체 1개월 백테스트.
    반환: 텔레그램 전송용 결과 문자열
    """
    if stocks is None:
        from config import STOCKS
        stocks = STOCKS

    logger.info("=== 1개월 ML 백테스트 시작 (%d종목) ===", len(stocks))
    results = []
    for ticker, name in stocks.items():
        r = backtest_ticker(ticker, name)
        if r:
            results.append(r)

    if not results:
        return "⚠️ 백테스트 결과 없음 — 데이터 또는 신호 부족"

    active  = [r for r in results if r.get("trades", 0) > 0]
    no_sig  = [r for r in results if r.get("trades", 0) == 0]

    lines = [
        f"📊 <b>1개월 ML 백테스트 결과</b>",
        f"기간: {date.today() - relativedelta(months=1)} ~ {date.today()}",
        f"대상: {len(stocks)}종목 / 신호 발생: {len(active)}종목\n",
    ]

    for r in sorted(active, key=lambda x: x.get("win_rate", 0), reverse=True):
        lines.append(
            f"<b>{r['name']}</b> ({r['ticker']})\n"
            f"  거래 {r['trades']}건 | 승률 {r['win_rate']:.1f}% | "
            f"손익비 {r['rr']:.2f} | 평균 {r['avg_pnl']:+.2f}%"
        )

    if no_sig:
        lines.append(f"\n신호 없음: {', '.join(r['name'] for r in no_sig)}")

    # 전체 종합
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
    from config import STOCKS
    print(run_backtest(STOCKS))
