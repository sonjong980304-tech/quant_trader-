"""
data_fetcher.py - yfinance를 이용한 주가 데이터 수집
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from config import STOCKS, BACKTEST_PERIOD_YEARS


def fetch_ohlcv(ticker: str, period_years: int = 1) -> pd.DataFrame:
    """
    종목 티커와 기간(년)을 받아 OHLCV 데이터프레임 반환.
    컬럼: Open, High, Low, Close, Volume
    """
    end   = datetime.today()
    start = end - timedelta(days=period_years * 365)

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"[{ticker}] 데이터를 가져올 수 없습니다.")

    # 멀티인덱스 컬럼 평탄화 (yfinance 0.2+ 대응)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def fetch_all_stocks(period_years: int = 1) -> dict:
    """
    config.STOCKS에 정의된 모든 종목의 OHLCV 데이터를 딕셔너리로 반환.
    반환: {ticker: DataFrame}
    """
    result = {}
    for ticker, name in STOCKS.items():
        try:
            df = fetch_ohlcv(ticker, period_years)
            result[ticker] = df
            print(f"  ✓ {name}({ticker}) {len(df)}일치 데이터 수집 완료")
        except Exception as e:
            print(f"  ✗ {name}({ticker}) 수집 실패: {e}")
    return result


if __name__ == "__main__":
    print("=== 데이터 수집 테스트 ===")
    data = fetch_all_stocks(period_years=1)
    for ticker, df in data.items():
        print(f"{ticker}: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}행")
