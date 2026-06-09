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


def retrain_daily(market: str = "all") -> dict:
    """
    유니버스 종목 병렬 재학습.
    market: 'kr' (KRX만) | 'us' (US만) | 'all' (둘 다)
    반환: {ticker: metrics or None}
    """
    from concurrent.futures import ThreadPoolExecutor
    from ml.model import train

    tickers_dict: dict = {}

    # 재학습은 과거 5년치 일봉 기반 → 실시간 스크리닝 불필요
    # 시가총액 기준 정적 유니버스 사용 (장외 시간에도 안정적으로 동작)
    if market in ("kr", "all"):
        try:
            from signals.krx_universe import get_krx_backtest_universe
            kr = get_krx_backtest_universe(top_n=200)
            if kr:
                tickers_dict.update(kr)
                logger.info("KRX 유니버스: %d개", len(kr))
            else:
                raise ValueError("KRX 유니버스 0개")
        except Exception as e:
            logger.warning("KRX 유니버스 조회 실패(%s) → config.STOCKS fallback", e)
            from config import STOCKS
            tickers_dict.update(STOCKS)
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
    # yfinance는 멀티스레드 환경에서 데이터가 섞이는 버그가 있으므로
    # 다운로드는 반드시 순서대로 처리
    logger.info("1단계: 데이터 다운로드 (직렬)")
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            data[ticker] = fetch_5y(ticker)
        except Exception as e:
            logger.error("  [FAIL 데이터] %s: %s", ticker, e)

    logger.info("다운로드 완료: %d / %d개", len(data), len(tickers))

    # ── 2단계: 모델 학습 (병렬) ─────────────────────────────────────
    # 학습은 각 스레드가 독립된 DataFrame을 사용하므로 병렬 안전
    logger.info("2단계: 모델 학습 (병렬 8스레드)")

    def _train_one(ticker: str):
        df = data.get(ticker)
        if df is None:
            return ticker, None
        best_metrics = None
        for agent in ("momentum", "reversion"):
            try:
                _, metrics = train(df, ticker, agent=agent)
                logger.info("  [OK] %s [%s] acc=%.3f auc=%.3f",
                            ticker, agent, metrics["accuracy"], metrics["auc"])
                if best_metrics is None or metrics["auc"] > best_metrics["auc"]:
                    best_metrics = metrics
            except Exception as e:
                logger.warning("  [SKIP] %s [%s]: %s", ticker, agent, e)
        if best_metrics is None:
            logger.error("  [FAIL] %s 두 에이전트 모두 학습 실패", ticker)
        return ticker, best_metrics

    results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for ticker, metrics in pool.map(_train_one, list(data.keys())):
            results[ticker] = metrics

    # 다운로드 실패 종목도 결과에 포함
    for ticker in tickers:
        if ticker not in results:
            results[ticker] = None

    ok   = sum(1 for v in results.values() if v)
    fail = len(results) - ok
    logger.info("일일 재학습 완료: 성공 %d / 실패 %d", ok, fail)
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
