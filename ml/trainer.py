from __future__ import annotations

"""
trainer.py - 관심종목 전체 XGBoost 모델 일괄 학습

사용법:
  python -m ml.trainer                   # 전체 STOCKS 학습
  python -m ml.trainer 005930.KS AAPL   # 특정 종목만 학습

학습 데이터: 최근 10년치 일봉 (yfinance)
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


def retrain_daily() -> dict:
    """
    매일 07:30 실행 — 전체 관심 종목을 최근 2년 데이터로 재학습.
    반환: {ticker: metrics or None}
    """
    from config import STOCKS, US_STOCKS
    from ml.model import train

    tickers = list(dict.fromkeys(list(STOCKS.keys()) + list(US_STOCKS.keys())))
    logger.info("일일 재학습 시작: %d개 종목", len(tickers))
    results = {}
    for ticker in tickers:
        try:
            df = fetch_5y(ticker)
            _, metrics = train(df, ticker)
            results[ticker] = metrics
            logger.info("  [OK] %s acc=%.3f auc=%.3f", ticker,
                        metrics["accuracy"], metrics["auc"])
        except Exception as e:
            logger.error("  [FAIL] %s: %s", ticker, e)
            results[ticker] = None
    ok   = sum(1 for v in results.values() if v)
    fail = len(results) - ok
    logger.info("일일 재학습 완료: 성공 %d / 실패 %d", ok, fail)
    return results


def train_ticker(ticker: str) -> dict | None:
    """단일 종목 학습. 성공 시 metrics 반환, 실패 시 None."""
    from ml.model import train
    try:
        df = fetch_10y(ticker)
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


if __name__ == "__main__":
    # CLI 인자로 특정 종목 지정 가능
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
    else:
        # 기본: config의 STOCKS + 안전자산
        from config import STOCKS
        from portfolio.safe_portfolio import SAFE_WEIGHTS
        tickers = list(STOCKS.keys()) + list(SAFE_WEIGHTS.keys())
        tickers = list(dict.fromkeys(tickers))  # 중복 제거

    logger.info("총 %d개 종목 학습 시작", len(tickers))
    results = train_all(tickers)
    print_summary(results)
