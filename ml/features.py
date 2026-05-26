from __future__ import annotations

"""
features.py - XGBoost 모델용 피처 엔지니어링

피처:
  - 기본: 변화율, 거래량 변화율
  - RSI (14일)
  - EMA 이격도 (20일 지수이동평균 대비)
  - 볼린저밴드 (20일 이동평균, 표준편차, 상/하한선, 폭, %B)
  - 거래량 비율 (20일 평균 대비)
  - 캔들 형태 (몸통, 위/아래꼬리)
  - 모멘텀 수익률 (3/5/10일)
  - 변동성 (10일 표준편차)
"""

import pandas as pd
import numpy as np


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(
    df: pd.DataFrame,
    ema_period: int = 20,
    bb_period: int = 20,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """
    OHLCV 데이터프레임에 ML 피처 컬럼 추가.
    입력 df 컬럼: Open, High, Low, Close, Volume
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # 중복 컬럼 제거 (yfinance MultiIndex 잔재)
    df = df.loc[:, ~df.columns.duplicated()]

    # ── 기본 수익률 / 거래량 ──────────────────────────
    df["change_rate"]    = df["Close"].squeeze().pct_change()
    df["volume_change"]  = df["Volume"].squeeze().pct_change()

    # ── RSI ──────────────────────────────────────────
    df["rsi"] = compute_rsi(df["Close"], period=rsi_period)

    # ── EMA & 이격도 ──────────────────────────────────
    ema_col       = f"ema_{ema_period}"
    dev_col       = f"ema_deviation_{ema_period}"
    df[ema_col]   = df["Close"].ewm(span=ema_period, adjust=False).mean()
    df[dev_col]   = (df["Close"] - df[ema_col]) / df[ema_col]

    # ── 볼린저밴드 ────────────────────────────────────
    bb_mid   = df["Close"].rolling(bb_period).mean()
    bb_std   = df["Close"].rolling(bb_period).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    df[f"bb_mid_{bb_period}"]   = bb_mid
    df[f"bb_std_{bb_period}"]   = bb_std
    df[f"bb_upper_{bb_period}"] = bb_upper
    df[f"bb_lower_{bb_period}"] = bb_lower
    df[f"bb_width_{bb_period}"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    pct_b_denom                 = (bb_upper - bb_lower).replace(0, np.nan)
    df[f"bb_pct_{bb_period}"]   = (df["Close"] - bb_lower) / pct_b_denom

    # ── 거래량 비율 ───────────────────────────────────
    vol_ma               = df["Volume"].rolling(20).mean()
    df["volume_ma20"]    = vol_ma
    df["volume_ratio"]   = df["Volume"] / vol_ma.replace(0, np.nan)

    # ── 캔들 형태 ─────────────────────────────────────
    open_safe            = df["Open"].replace(0, np.nan)
    candle_top           = df[["Open", "Close"]].max(axis=1)
    candle_bottom        = df[["Open", "Close"]].min(axis=1)
    df["candle_body"]        = (df["Close"] - df["Open"]) / open_safe
    df["candle_upper_wick"]  = (df["High"]  - candle_top)    / open_safe
    df["candle_lower_wick"]  = (candle_bottom - df["Low"])   / open_safe

    # ── 모멘텀 수익률 ─────────────────────────────────
    for w in [3, 5, 10]:
        df[f"ret_{w}d"] = df["Close"].pct_change(w)

    # ── 변동성 ───────────────────────────────────────
    df["volatility_10d"] = df["change_rate"].rolling(10).std()

    # inf 값은 dropna로 걸러지지 않으므로 NaN으로 치환
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# 모델에 사용할 피처 컬럼 목록
FEATURE_COLS = [
    "change_rate",
    "volume_change",
    "rsi",
    "ema_deviation_20",
    "bb_width_20",
    "bb_pct_20",
    "bb_std_20",
    "volume_ratio",
    "candle_body",
    "candle_upper_wick",
    "candle_lower_wick",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "volatility_10d",
]


def make_target(
    df: pd.DataFrame,
    horizon: int = 7,
    threshold: float = 0.03,
) -> tuple[pd.Series, pd.Series]:
    """
    7일 후 수익률 기반 이진 분류 타겟 생성.

    threshold : 이 수익률 이상이면 성공(1), 미만이면 실패(0)
    반환: (labels, future_returns)
    """
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    labels        = (future_return >= threshold).astype(int)
    return labels, future_return
