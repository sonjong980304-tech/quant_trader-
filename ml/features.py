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

    # ── ATR (14일) ────────────────────────────────────
    prev_close    = df["Close"].shift(1)
    tr            = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"]  = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["Close"].replace(0, np.nan)  # 가격 정규화

    # ── MA200 (추세 필터용, 피처 아님) ───────────────────
    df["ma200"] = df["Close"].rolling(200).mean()

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
    "atr_pct",             # Phase 4: ATR 정규화 (변동성 인식)
]


def detect_momentum_rows(df: pd.DataFrame) -> pd.Series:
    """돌파 에이전트 학습용: ① 거래량폭발 OR ⑤ BB스퀴즈돌파 조건이 발생한 행."""
    if "volume_ratio" not in df.columns:
        df = add_features(df)

    # ① 거래량폭발: volume_ratio > 1.5 + 양봉 (학습용은 1.5로 완화 — 2.0은 데이터 부족)
    vol_explode = (df["volume_ratio"] > 1.5) & (df["candle_body"] > 0)

    # ⑤ BB스퀴즈돌파: 전일 BB폭이 60일 최저 근처 + 금일 종가가 BB상단 돌파
    bb_upper = df["bb_mid_20"] + 2 * df["bb_std_20"]
    min_w60  = df["bb_width_20"].rolling(60).min().shift(1)
    bb_squeeze = (df["bb_width_20"].shift(1) <= min_w60 * 1.1) & (df["Close"] > bb_upper)

    return (vol_explode | bb_squeeze).fillna(False)


def detect_reversion_rows(df: pd.DataFrame) -> pd.Series:
    """눌림목 에이전트 학습용: ② BB하단반등 OR ③ RSI과매도탈출 OR ④ 이격도저점 조건이 발생한 행."""
    if "volume_ratio" not in df.columns:
        df = add_features(df)

    # ② BB하단반등: 전일 bb_pct < 0 → 금일 bb_pct ≥ 0
    bb_bounce = (df["bb_pct_20"].shift(1) < 0) & (df["bb_pct_20"] >= 0)

    # ③ RSI과매도탈출: 전일 rsi < 30 → 금일 rsi ≥ 30
    rsi_escape = (df["rsi"].shift(1) < 30) & (df["rsi"] >= 30)

    # ④ 이격도저점: ema_deviation_20 ≤ -5%
    ema_low = df["ema_deviation_20"] <= -0.05

    return (bb_bounce | rsi_escape | ema_low).fillna(False)


def make_target(
    df: pd.DataFrame,
    horizon: int = 7,
    threshold: float = 0.03,
) -> tuple[pd.Series, pd.Series]:
    """
    [레거시] 7일 후 종가 수익률 기반 이진 분류 타겟.
    신규 학습은 triple_barrier_label() 사용 권장.
    """
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    labels        = (future_return >= threshold).astype(int)
    return labels, future_return


def triple_barrier_label(
    df: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
    max_holding_days: int = 7,
    use_intraday: bool = True,
) -> pd.Series:
    """
    López de Prado식 Triple-Barrier 라벨링.

    배리어:
      상단(TP): entry_price * (1 + tp_pct)  → 먼저 도달 시 label=1
      하단(SL): entry_price * (1 - sl_pct)  → 먼저 도달 시 label=0
      시간(vertical): max_holding_days 경과 후 종가 ≥ 진입가 → 1, 아니면 0

    NOTE: 같은 날 TP·SL 동시 터치 → SL 우선.
          실거래에서 최악의 경우를 가정해 라벨 낙관 편향을 방지.

    Args:
        df               : OHLCV 데이터프레임 (거래일 인덱스)
        tp_pct           : 익절 기준 (예: 0.07 → +7%)
        sl_pct           : 손절 기준, 양수 전달 (예: 0.07 → -7%)
        max_holding_days : 최대 보유 거래일 수
        use_intraday     : True → 장중 High/Low로 판정 (기본값)
                           False → 종가만 사용 (단위 테스트 비교용)

    Returns:
        pd.Series: index=진입 시점, value ∈ {0, 1}, 마지막 horizon행은 NaN
    """
    close = df["Close"].squeeze().values.astype(float)
    high  = df["High"].squeeze().values.astype(float) if use_intraday else close
    low   = df["Low"].squeeze().values.astype(float)  if use_intraday else close
    n     = len(df)

    labels = np.full(n, np.nan)

    for i in range(n):
        entry = close[i]
        if entry <= 0 or np.isnan(entry):
            continue
        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 - sl_pct)

        label       = np.nan
        horizon_end = min(i + max_holding_days + 1, n)

        for j in range(i + 1, horizon_end):
            hit_sl = low[j]  <= sl_price
            hit_tp = high[j] >= tp_price
            # 같은 날 동시 터치 → SL 우선 (보수적 가정)
            if hit_sl:
                label = 0.0
                break
            if hit_tp:
                label = 1.0
                break

        # 시간 배리어: 만료 시점 종가 기준
        if np.isnan(label):
            final_idx = min(i + max_holding_days, n - 1)
            label = 1.0 if close[final_idx] >= entry else 0.0

        labels[i] = label

    return pd.Series(labels, index=df.index, dtype="float64")


def _triple_barrier_pnl(
    df: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
    max_holding_days: int = 7,
    cost_pct: float = 0.0046,
) -> tuple[pd.Series, pd.Series]:
    """
    Triple-Barrier 라벨 + 실제 손익률 동시 반환 (model.py 내부용).

    손익률:
      TP 도달  → +tp_pct
      SL 도달  → -sl_pct
      시간 만료 → 실제 종가 수익률

    cost_pct: 왕복 비용 (수수료 0.03% + 슬리피지 0.25% + 증권거래세 0.18% = 0.46%).
              수직 배리어 label 기준: pnl >= cost_pct 이어야 win(1).
              TP/SL 배리어는 가격 경계로 판정하므로 cost는 label이 아닌 net_pnl에만 반영.
    """
    close = df["Close"].squeeze().values.astype(float)
    high  = df["High"].squeeze().values.astype(float)
    low   = df["Low"].squeeze().values.astype(float)
    n     = len(df)

    labels  = np.full(n, np.nan)
    returns = np.full(n, np.nan)

    for i in range(n):
        entry = close[i]
        if entry <= 0 or np.isnan(entry):
            continue
        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 - sl_pct)

        label       = np.nan
        pnl         = np.nan
        horizon_end = min(i + max_holding_days + 1, n)

        for j in range(i + 1, horizon_end):
            if low[j] <= sl_price:
                label = 0.0
                pnl   = -sl_pct
                break
            if high[j] >= tp_price:
                label = 1.0
                pnl   = tp_pct
                break

        if np.isnan(label):
            final_idx = min(i + max_holding_days, n - 1)
            pnl   = (close[final_idx] - entry) / entry
            # 비용 차감 후 양수여야 win — 0.0 기준은 비용 편향을 유발
            label = 1.0 if pnl >= cost_pct else 0.0

        labels[i]  = label
        returns[i] = pnl

    return (
        pd.Series(labels,  index=df.index, dtype="float64"),
        pd.Series(returns, index=df.index, dtype="float64"),
    )
