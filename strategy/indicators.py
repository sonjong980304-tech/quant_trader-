"""
indicators.py - 기술적 지표 계산 (이동평균, RSI, 거래량 보정, MA20 방향)
ta 라이브러리 활용 (Python 3.9 호환)
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo-root: 하위 폴더에서 직접 실행 대비

import pandas as pd
import ta
from datetime import datetime
from config import (
    MA_LONG, MA20_RISING_LOOKBACK,
    MARKET_OPEN_HOUR, MARKET_OPEN_MIN, MARKET_TOTAL_MINUTES,
    VOLUME_INCREASE_RATIO, VOLUME_SURGE_RATIO,
)


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


# ─────────────────────────────────────────────
# MA20 방향 판단
# ─────────────────────────────────────────────

def is_ma20_rising(df: pd.DataFrame, lookback: int = MA20_RISING_LOOKBACK) -> bool:
    """
    최근 lookback일 동안 20일 이동평균선이 우상향인지 판단 (마지막 행 기준).
    ma20[오늘] > ma20[오늘 - lookback] 이면 True.
    """
    ma20_col = f"MA_{MA_LONG}"
    if ma20_col not in df.columns or len(df) < lookback + 1:
        return False
    return float(df[ma20_col].iloc[-1]) > float(df[ma20_col].iloc[-1 - lookback])


# ─────────────────────────────────────────────
# 거래량 시간대 보정
# ─────────────────────────────────────────────

def get_elapsed_ratio(now: datetime = None) -> float:
    """
    장 시작(9:00) 기준 경과 비율 반환 (0.0 ~ 1.0).
    9:30 이전이면 None 반환 → 호출자가 신호 제외 처리.

    시간대별 신뢰도:
      09:00~09:30 : 신호 제외 (변동성 극심)
      09:30~10:30 : 경과비율 보정 필수
      10:30~14:00 : 신뢰도 가장 높은 구간
      14:00~15:30 : 마감 거래량 몰림 주의
    """
    if now is None:
        now = datetime.now()
    open_time = now.replace(
        hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN,
        second=0, microsecond=0,
    )
    if now < open_time.replace(minute=30):
        return None
    elapsed = (now - open_time).total_seconds() / 60
    return min(elapsed / MARKET_TOTAL_MINUTES, 1.0)


def get_projected_volume(current_vol: int, now: datetime = None) -> float:
    """
    현재 누적 거래량을 하루 예상 거래량으로 환산.
    projected = current_vol / elapsed_ratio
    elapsed_ratio가 None(9:30 이전)이면 0.0 반환.
    """
    ratio = get_elapsed_ratio(now)
    if ratio is None or ratio < 0.001:
        return 0.0
    return current_vol / ratio


def is_volume_increasing(current_vol: int, avg_daily_vol: float, now: datetime = None) -> bool:
    """거래량 증가 여부: 예상 일거래량 > 평균 일거래량 × VOLUME_INCREASE_RATIO"""
    projected = get_projected_volume(current_vol, now)
    return projected > avg_daily_vol * VOLUME_INCREASE_RATIO


def is_volume_surge(current_vol: int, avg_daily_vol: float, now: datetime = None) -> bool:
    """거래량 급증 여부: 예상 일거래량 > 평균 일거래량 × VOLUME_SURGE_RATIO"""
    projected = get_projected_volume(current_vol, now)
    return projected > avg_daily_vol * VOLUME_SURGE_RATIO


def is_volume_decreasing_trend(volume_series: list, days: int = 3) -> bool:
    """
    최근 days일 거래량이 연속 감소 중인지 판단.
    volume_series는 최근순 정렬 (index 0이 가장 최근).
    """
    if len(volume_series) < days + 1:
        return False
    for i in range(days):
        if volume_series[i] >= volume_series[i + 1]:
            return False
    return True


if __name__ == "__main__":
    from data.data_fetcher import fetch_ohlcv
    df = fetch_ohlcv("005930.KS", period_years=1)
    df = add_all_indicators(df)
    df = detect_crossover(df)
    print(df[["Close", "MA_5", "MA_20", "RSI", "golden_cross", "dead_cross"]].tail(10))
    print(f"\nMA20 우상향: {is_ma20_rising(df)}")
    print(f"장중 경과 비율: {get_elapsed_ratio()}")
