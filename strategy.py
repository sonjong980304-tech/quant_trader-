"""
strategy.py - 매수/매도 신호 생성
골든크로스 + RSI 모멘텀 전략 구현
"""

import pandas as pd
from config import ACTIVE_STRATEGY


def generate_signals(df: pd.DataFrame, strategy: dict = None) -> pd.DataFrame:
    """
    전략 파라미터에 따라 매수/매도 신호 컬럼을 생성하여 반환.

    컬럼:
      buy_signal      : True = 매수 신호
      sell_full       : True = 전량 매도 (데드크로스)
      sell_partial    : True = 분할 매도 50% (RSI 과매수 후 하락)
      rsi_was_overbought: RSI가 75 이상을 찍은 뒤 추적용 플래그
    """
    if strategy is None:
        strategy = ACTIVE_STRATEGY

    s  = strategy["short_window"]
    l  = strategy["long_window"]
    rb = strategy["rsi_buy_threshold"]
    ro = strategy["rsi_overbought"]
    re = strategy["rsi_overbought_exit"]

    df = df.copy()

    # ── 매수 신호: 골든크로스 AND RSI >= 매수 임계값
    df["buy_signal"] = (
        df["golden_cross"] &
        (df["RSI"] >= rb)
    )

    # ── 전량 매도: 데드크로스
    df["sell_full"] = df["dead_cross"].copy()

    # ── 분할 매도: RSI가 75 이상 도달 후 70 밑으로 하락
    # rsi_was_overbought 플래그를 앞에서 뒤로 전파
    rsi_over = df["RSI"] >= ro
    rsi_over_any = rsi_over.copy().astype(bool)

    # 과매수 도달 여부를 누적 플래그로 변환 (데드크로스 발생 시 리셋은 간소화)
    df["rsi_was_overbought"] = False
    flag = False
    for i in range(len(df)):
        if df["RSI"].iloc[i] >= ro:
            flag = True
        if df["dead_cross"].iloc[i]:
            flag = False  # 데드크로스 발생 시 플래그 초기화
        df.iloc[i, df.columns.get_loc("rsi_was_overbought")] = flag

    df["sell_partial"] = (
        df["rsi_was_overbought"] &
        (df["RSI"] < re) &
        (df["RSI"].shift(1) >= re)  # 70 아래로 하락하는 시점
    )

    return df


def get_latest_signal(df: pd.DataFrame) -> dict:
    """
    최신 행의 신호를 딕셔너리로 반환.
    {
      "buy": bool,
      "sell_full": bool,
      "sell_partial": bool,
      "rsi": float,
      "ma_short": float,
      "ma_long": float,
      "date": str,
    }
    """
    last = df.iloc[-1]
    short = ACTIVE_STRATEGY["short_window"]
    long  = ACTIVE_STRATEGY["long_window"]

    return {
        "buy":          bool(last.get("buy_signal", False)),
        "sell_full":    bool(last.get("sell_full", False)),
        "sell_partial": bool(last.get("sell_partial", False)),
        "rsi":          round(float(last.get("RSI", 0)), 2),
        "ma_short":     round(float(last.get(f"MA_{short}", 0)), 2),
        "ma_long":      round(float(last.get(f"MA_{long}", 0)), 2),
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
    print("=== 최신 신호 ===")
    for k, v in sig.items():
        print(f"  {k}: {v}")
