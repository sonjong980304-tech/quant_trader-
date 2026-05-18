"""
indicators.py - 기술적 지표 계산 (이동평균, RSI)
ta 라이브러리 활용 (Python 3.9 호환)
"""

import pandas as pd
import ta


def add_moving_averages(df: pd.DataFrame, short: int = 5, long: int = 20) -> pd.DataFrame:
    """
    단기/장기 단순이동평균(SMA) 컬럼을 추가하여 반환.
    컬럼명: MA_{short}, MA_{long}
    """
    df = df.copy()
    df[f"MA_{short}"] = ta.trend.sma_indicator(df["Close"], window=short)
    df[f"MA_{long}"]  = ta.trend.sma_indicator(df["Close"], window=long)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI 컬럼을 추가하여 반환.
    컬럼명: RSI
    """
    df = df.copy()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=period).rsi()
    return df


def add_all_indicators(
    df: pd.DataFrame,
    short: int = 5,
    long: int = 20,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """이동평균 + RSI를 한 번에 추가하여 반환."""
    df = add_moving_averages(df, short, long)
    df = add_rsi(df, rsi_period)
    df.dropna(inplace=True)
    return df


def detect_crossover(df: pd.DataFrame, short: int = 5, long: int = 20) -> pd.DataFrame:
    """
    골든크로스 / 데드크로스 감지 컬럼 추가.
    golden_cross: 이전 MA_short <= MA_long AND 현재 MA_short > MA_long
    dead_cross:   이전 MA_short >= MA_long AND 현재 MA_short < MA_long
    """
    df = df.copy()
    ma_s = f"MA_{short}"
    ma_l = f"MA_{long}"

    df["golden_cross"] = (
        (df[ma_s].shift(1) <= df[ma_l].shift(1)) &
        (df[ma_s] > df[ma_l])
    )
    df["dead_cross"] = (
        (df[ma_s].shift(1) >= df[ma_l].shift(1)) &
        (df[ma_s] < df[ma_l])
    )
    return df


if __name__ == "__main__":
    from data_fetcher import fetch_ohlcv
    df = fetch_ohlcv("005930.KS", period_years=1)
    df = add_all_indicators(df)
    df = detect_crossover(df)
    print(df[["Close", "MA_5", "MA_20", "RSI", "golden_cross", "dead_cross"]].tail(10))
