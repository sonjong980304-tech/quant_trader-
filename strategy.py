"""
strategy.py - 매수/매도 신호 생성
5일/20일 이동평균선 + 거래량 + 캔들 타입 + MA20 방향 필터 기반 전략
"""

import pandas as pd
from config import (
    MA_SHORT, MA_LONG, MA20_RISING_LOOKBACK,
    VOLUME_INCREASE_RATIO, VOLUME_SURGE_RATIO,
    VOLUME_LOOKBACK_DAYS, DOJI_THRESHOLD,
    LARGE_CANDLE_MULTIPLIER, PULLBACK_DAYS,
)


# ─────────────────────────────────────────────
# 캔들 판단 함수
# ─────────────────────────────────────────────

def _body(df: pd.DataFrame) -> pd.Series:
    return df["Close"] - df["Open"]


def _is_bullish(df: pd.DataFrame) -> pd.Series:
    """양봉: 종가 > 시가"""
    return _body(df) > 0


def _is_bearish(df: pd.DataFrame) -> pd.Series:
    """음봉: 종가 < 시가"""
    return _body(df) < 0


def _is_doji(df: pd.DataFrame) -> pd.Series:
    """도지: 몸통 / 전체범위 < DOJI_THRESHOLD"""
    total_range = (df["High"] - df["Low"]).replace(0, float("nan"))
    return (_body(df).abs() / total_range) < DOJI_THRESHOLD


def _is_large_bearish(df: pd.DataFrame) -> pd.Series:
    """장대음봉: 음봉이면서 몸통 > 직전 5일 평균 캔들 크기 × LARGE_CANDLE_MULTIPLIER"""
    body     = _body(df)
    avg_body = body.abs().rolling(window=5, min_periods=1).mean().shift(1)
    return (body < 0) & (body.abs() > avg_body * LARGE_CANDLE_MULTIPLIER)


# ─────────────────────────────────────────────
# 거래량 판단 함수 (일봉 기반)
# ─────────────────────────────────────────────

def _vol_ma(df: pd.DataFrame) -> pd.Series:
    """직전 VOLUME_LOOKBACK_DAYS일 평균 거래량 (오늘 제외)"""
    return df["Volume"].rolling(window=VOLUME_LOOKBACK_DAYS, min_periods=1).mean().shift(1)


def _projected_volume(df: pd.DataFrame) -> pd.Series:
    """
    마지막 행이 오늘 날짜인 경우 현재까지의 거래량을 하루 예상 거래량으로 환산.
    - 9:30 이전: 마지막 행 거래량을 0으로 처리 → 신호 미발생
    - 9:30 이후 장중: current_vol / elapsed_ratio 로 환산
    - 백테스트(과거 데이터) 또는 장 종료 후: 원본 거래량 그대로 반환
    """
    import datetime
    from indicators import get_elapsed_ratio

    vol = df["Volume"].copy().astype(float)

    last_date = df.index[-1]
    if hasattr(last_date, "date"):
        last_date = last_date.date()

    if last_date == datetime.date.today():
        ratio = get_elapsed_ratio()
        if ratio is None:
            vol.iloc[-1] = 0.0
        else:
            vol.iloc[-1] = vol.iloc[-1] / ratio

    return vol


def _vol_increase(df: pd.DataFrame) -> pd.Series:
    """동시간대 예상 일거래량 > 직전 N일 평균 × VOLUME_INCREASE_RATIO"""
    return _projected_volume(df) > _vol_ma(df) * VOLUME_INCREASE_RATIO


def _vol_surge(df: pd.DataFrame) -> pd.Series:
    """동시간대 예상 일거래량 > 직전 N일 평균 × VOLUME_SURGE_RATIO"""
    return _projected_volume(df) > _vol_ma(df) * VOLUME_SURGE_RATIO


def _vol_decreasing_n_days(df: pd.DataFrame, n: int) -> pd.Series:
    """직전 n일간 거래량이 매일 감소하는지 확인"""
    result = pd.Series(True, index=df.index)
    for i in range(1, n + 1):
        result &= df["Volume"].shift(i) < df["Volume"].shift(i + 1)
    return result


def _bearish_n_days(df: pd.DataFrame, n: int) -> pd.Series:
    """직전 n일 연속 음봉인지 확인"""
    result = pd.Series(True, index=df.index)
    for i in range(1, n + 1):
        result &= df["Close"].shift(i) < df["Open"].shift(i)
    return result


def _ma20_rising_series(df: pd.DataFrame) -> pd.Series:
    """
    MA20 우상향 판단 (Series 버전 — 백테스트/신호 생성용).
    ma20[t] > ma20[t - MA20_RISING_LOOKBACK] 이면 True.
    """
    ma20_col = f"MA_{MA_LONG}"
    return df[ma20_col] > df[ma20_col].shift(MA20_RISING_LOOKBACK)


# ─────────────────────────────────────────────
# 매수 신호 (일봉 기반)
# ─────────────────────────────────────────────

def buy_signal_1(df: pd.DataFrame) -> pd.Series:
    """
    매수 1원칙: 5일선 위 단기 돌파 + MA20 우상향 필터
    - MA20 우상향
    - 전일 종가 > MA5 (5일선 위에 있는 상태)
    - 당일 시가 < MA5 (5일선 아래로 하락 출발)
    - 당일 종가 > MA5 + 양봉 (5일선 위로 재돌파)
    """
    ma5 = f"MA_{MA_SHORT}"
    return (
        _ma20_rising_series(df) &
        (df["Close"].shift(1) > df[ma5].shift(1)) &
        (df["Open"] < df[ma5]) &
        _is_bullish(df) &
        (df["Close"] > df[ma5])
    )


def buy_signal_2(df: pd.DataFrame) -> pd.Series:
    """
    매수 2원칙: 5~20일선 사이 반등 + MA20 우상향 필터
    - MA20 우상향
    - 종가가 MA5와 MA20 사이
    - 오늘 거래량 증가 + 양봉
    """
    ma5  = f"MA_{MA_SHORT}"
    ma20 = f"MA_{MA_LONG}"
    ma_min = df[[ma5, ma20]].min(axis=1)
    ma_max = df[[ma5, ma20]].max(axis=1)

    between_mas    = (df["Close"] > ma_min) & (df["Close"] < ma_max)
    today_reversal = _vol_increase(df) & _is_bullish(df)

    return _ma20_rising_series(df) & between_mas & today_reversal


def buy_signal_3(df: pd.DataFrame) -> pd.Series:
    """
    매수 3원칙: 거래량 급증 반등 + MA20 우상향 필터
    - MA20 우상향
    - 거래량 급증
    - 양봉 또는 도지형 캔들
    """
    return (
        _ma20_rising_series(df) &
        _vol_surge(df) &
        (_is_bullish(df) | _is_doji(df))
    )


# ─────────────────────────────────────────────
# 매도 신호
# ─────────────────────────────────────────────

def sell_signal_1(df: pd.DataFrame) -> pd.Series:
    """
    매도 1원칙: 5일선 위 급등 후 장대음봉
    - 종가 > MA5
    - 거래량 급증 + 장대음봉
    """
    ma5 = f"MA_{MA_SHORT}"
    return (
        (df["Close"] > df[ma5]) &
        _vol_surge(df) &
        _is_large_bearish(df)
    )


def sell_signal_2(df: pd.DataFrame) -> pd.Series:
    """
    매도 2원칙: 5~20일선 사이 거래량 증가 음봉
    - 종가가 MA5와 MA20 사이
    - 거래량 증가 + 음봉
    """
    ma5  = f"MA_{MA_SHORT}"
    ma20 = f"MA_{MA_LONG}"
    ma_min = df[[ma5, ma20]].min(axis=1)
    ma_max = df[[ma5, ma20]].max(axis=1)

    return (
        (df["Close"] > ma_min) &
        (df["Close"] < ma_max) &
        _vol_increase(df) &
        _is_bearish(df)
    )


# ─────────────────────────────────────────────
# 분봉 기반 매수 1원칙 (실시간 장중 전용)
# ─────────────────────────────────────────────

def buy_signal_1_intraday(ticker: str, minute_df: pd.DataFrame, daily_df: pd.DataFrame) -> bool:
    """
    분봉 기반 매수 1원칙 (runner.py 장중 실행 전용).

    조건:
      1. MA20 우상향 필터 (일봉 기준)
      2. 현재가 > 5일 이동평균선 (일봉 기준)
      3. 9:00~9:30 사이 저가가 5일선 아래로 내려간 적 있을 것
      4. 현재가가 장 시작 시가(9:00 첫 캔들 open)를 상향 돌파
      5. 돌파 캔들 거래량 > 직전 5분봉 평균거래량 × VOLUME_INCREASE_RATIO
    """
    from indicators import is_ma20_rising

    if minute_df.empty or daily_df.empty:
        return False

    # 1. MA20 우상향 필터
    if not is_ma20_rising(daily_df):
        return False

    ma5_col   = f"MA_{MA_SHORT}"
    ma5_value = float(daily_df[ma5_col].iloc[-1])

    # 2. 현재가 > MA5
    current_price = float(minute_df["close"].iloc[-1])
    if current_price <= ma5_value:
        return False

    # 분봉 인덱스 DatetimeIndex 보장
    if not isinstance(minute_df.index, pd.DatetimeIndex):
        minute_df = minute_df.copy()
        minute_df.index = pd.to_datetime(minute_df.index)

    # 3. 9:00~9:30 구간에 저가가 MA5 아래로 내려간 적 있는지
    try:
        early_df = minute_df.between_time("09:00", "09:30")
    except Exception:
        return False

    if early_df.empty:
        return False

    if not (early_df["low"] < ma5_value).any():
        return False

    # 4. 현재 캔들이 장 시작 시가를 상향 돌파
    if len(minute_df) < 2:
        return False

    open_price    = float(minute_df["open"].iloc[0])   # 9:00 첫 캔들 시가
    prev_close    = float(minute_df["close"].iloc[-2])
    current_close = float(minute_df["close"].iloc[-1])

    if not (prev_close <= open_price < current_close):
        return False

    # 5. 거래량 조건: 현재 캔들 거래량 > 직전 5분봉 평균 × VOLUME_INCREASE_RATIO
    if len(minute_df) < 6:
        return False

    current_vol = int(minute_df["volume"].iloc[-1])
    avg_vol_5   = float(minute_df["volume"].iloc[-6:-1].mean())

    if avg_vol_5 <= 0:
        return False

    return current_vol > avg_vol_5 * VOLUME_INCREASE_RATIO


# ─────────────────────────────────────────────
# 통합 신호 생성
# ─────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, strategy: dict = None) -> pd.DataFrame:
    """
    전략 신호 컬럼 생성.
    strategy 파라미터는 하위 호환성을 위해 유지하되 무시됨.

    추가 컬럼:
      buy_signal_1/2/3, sell_signal_1/2
      buy_signal    : 매수 1~3원칙 중 하나라도 해당
      sell_full     : 매도 1원칙 (즉시 전량 매도)
      sell_partial  : 매도 2원칙 (분할 매도)
    """
    df = df.copy()

    df["buy_signal_1"]  = buy_signal_1(df)
    df["buy_signal_2"]  = buy_signal_2(df)
    df["buy_signal_3"]  = buy_signal_3(df)
    df["sell_signal_1"] = sell_signal_1(df)
    df["sell_signal_2"] = sell_signal_2(df)

    df["buy_signal"]   = df["buy_signal_1"] | df["buy_signal_2"] | df["buy_signal_3"]
    df["sell_full"]    = df["sell_signal_1"] | df["sell_signal_2"]
    df["sell_partial"] = pd.Series(False, index=df.index)

    return df


def get_latest_signal(df: pd.DataFrame) -> dict:
    """최신 행의 신호를 딕셔너리로 반환"""
    last = df.iloc[-1]

    buy_which = []
    if last.get("buy_signal_1", False):  buy_which.append("1원칙")
    if last.get("buy_signal_2", False):  buy_which.append("2원칙")
    if last.get("buy_signal_3", False):  buy_which.append("3원칙")

    sell_which = []
    if last.get("sell_signal_1", False): sell_which.append("1원칙")
    if last.get("sell_signal_2", False): sell_which.append("2원칙")

    return {
        "buy":          bool(last.get("buy_signal", False)),
        "sell_full":    bool(last.get("sell_full", False)),
        "sell_partial": bool(last.get("sell_partial", False)),
        "buy_which":    buy_which,
        "sell_which":   sell_which,
        "rsi":          round(float(last.get("RSI", 0)), 2),
        "ma_short":     round(float(last.get(f"MA_{MA_SHORT}", 0)), 2),
        "ma_long":      round(float(last.get(f"MA_{MA_LONG}", 0)), 2),
        "close":        round(float(last.get("Close", 0)), 0),
        "volume":       int(last.get("Volume", 0)),
        "date":         str(df.index[-1].date()),
    }


if __name__ == "__main__":
    from data_fetcher import fetch_ohlcv
    from indicators import add_all_indicators, detect_crossover

    df = fetch_ohlcv("005930.KS", period_years=1)
    df = add_all_indicators(df)
    df = detect_crossover(df)
    df = generate_signals(df)

    sig = get_latest_signal(df)
    print("=== 최신 신호 (삼성전자) ===")
    for k, v in sig.items():
        print(f"  {k}: {v}")

    print("\n=== 신호 발생 건수 (1년) ===")
    print(f"  매수(1원칙): {df['buy_signal_1'].sum()}건")
    print(f"  매수(2원칙): {df['buy_signal_2'].sum()}건")
    print(f"  매수(3원칙): {df['buy_signal_3'].sum()}건")
    print(f"  매도(1원칙): {df['sell_signal_1'].sum()}건")
    print(f"  매도(2원칙): {df['sell_signal_2'].sum()}건")
