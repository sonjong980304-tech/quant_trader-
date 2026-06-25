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
    kospi_df: pd.DataFrame | None = None,
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

    # ── 거래량 비율 / 추세 ──────────────────────────────
    vol_ma               = df["Volume"].rolling(20).mean()
    df["volume_ma20"]    = vol_ma
    df["volume_ratio"]   = df["Volume"] / vol_ma.replace(0, np.nan)
    vol_5d_ma            = df["Volume"].rolling(5).mean()
    df["turnover_trend"] = vol_5d_ma / vol_ma.replace(0, np.nan)
    df["volume_surge_5d"] = df["turnover_trend"]  # 별칭

    # ── 모멘텀 변동성 (20일 일간수익률 표준편차) ─────────
    df["mom_volatility"] = df["Close"].pct_change().rolling(20).std()

    # ── 섹터 모멘텀 (gridsearch에서 합산 후 덮어씀) ──────
    df["sector_momentum_5d"] = np.nan

    # ── 캔들 형태 ─────────────────────────────────────
    open_safe            = df["Open"].replace(0, np.nan)
    candle_top           = df[["Open", "Close"]].max(axis=1)
    candle_bottom        = df[["Open", "Close"]].min(axis=1)
    df["candle_body"]        = (df["Close"] - df["Open"]) / open_safe
    df["candle_upper_wick"]  = (df["High"]  - candle_top)    / open_safe
    df["candle_lower_wick"]  = (candle_bottom - df["Low"])   / open_safe

    # ── 모멘텀 수익률 ─────────────────────────────────
    for w in [3, 5, 10, 20, 60]:
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
    df["atr_pct"] = df["atr_14"] / df["Close"].replace(0, np.nan)

    # ── 52주 고저 이격도 ──────────────────────────────
    high52         = df["Close"].rolling(252, min_periods=200).max()
    low52          = df["Close"].rolling(252, min_periods=200).min()
    df["high52_pct"] = (df["Close"] - high52) / high52.replace(0, np.nan)
    df["low52_pct"]  = (df["Close"] - low52)  / low52.replace(0, np.nan)

    # ── OBV 변화율 (5일) ─────────────────────────────
    close_sq  = df["Close"].squeeze()
    vol_sq    = df["Volume"].squeeze()
    obv       = (vol_sq * np.sign(close_sq - close_sq.shift(1))).fillna(0).cumsum()
    df["obv"] = obv
    obv_5d_ago          = obv.shift(5)
    df["obv_change_5d"] = (obv - obv_5d_ago) / (obv_5d_ago.abs() + 1e-9)

    # ── RSI 과매도 깊이 ───────────────────────────────
    df["rsi_oversold"] = (30 - df["rsi"]).clip(lower=0)

    # ── KOSPI 상대 수익률 ────────────────────────────
    if kospi_df is not None:
        try:
            kospi_close  = kospi_df["Close"].squeeze()
            kospi_ret5   = kospi_close.pct_change(5).reindex(df.index, method="ffill")
            kospi_ret20  = kospi_close.pct_change(20).reindex(df.index, method="ffill")
            df["kospi_relative_5d"]  = df["ret_5d"]  - kospi_ret5
            df["kospi_relative_20d"] = df["ret_20d"] - kospi_ret20
            # beta_60d: 60일 롤링 KOSPI 대비 베타
            kospi_daily = kospi_close.pct_change().reindex(df.index, method="ffill")
            stock_daily = df["Close"].pct_change()
            cov60 = stock_daily.rolling(60).cov(kospi_daily)
            var60 = kospi_daily.rolling(60).var()
            df["beta_60d"] = cov60 / var60.replace(0, np.nan)
        except Exception:
            df["kospi_relative_5d"]  = np.nan
            df["kospi_relative_20d"] = np.nan
            df["beta_60d"]           = np.nan
    else:
        df["kospi_relative_5d"]  = np.nan
        df["kospi_relative_20d"] = np.nan
        df["beta_60d"]           = np.nan

    # ── 이동평균 및 눌림목 피처 ──────────────────────────
    _c            = df["Close"]
    _ma5          = _c.rolling(5).mean()
    _ma20         = _c.rolling(20).mean()
    _ma60         = _c.rolling(60).mean()
    _ma200        = _c.rolling(200).mean()
    df["ma200"]           = _ma200
    df["ma200_deviation"] = (_c - _ma200) / _ma200.replace(0, np.nan)
    df["ma20_deviation"]  = (_c - _ma20)  / _ma20.replace(0, np.nan)
    df["ma_alignment"]    = ((_ma5 > _ma20) & (_ma20 > _ma60)).astype(float)
    _high20               = _c.rolling(20).max()
    df["pullback_depth"]  = (_high20 - _c) / _high20.replace(0, np.nan)
    # ma20_cross: 최근 3일 내 MA20 상향 돌파 (0/1)
    _cross = (_c.shift(1) < _ma20.shift(1)) & (_c >= _ma20)
    df["ma20_cross"] = _cross.rolling(3).max().astype(float)

    # inf 값은 dropna로 걸러지지 않으므로 NaN으로 치환
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# ── 전역 momentum 모델 피처 (17개, 레거시) ─────────────────────────────────
FEATURE_COLS = [
    "change_rate",
    "volume_change",
    "rsi",
    "ema_deviation_20",
    "bb_pct_20",
    "bb_std_20",
    "volume_ratio",
    "candle_body",
    "candle_upper_wick",
    "candle_lower_wick",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "atr_pct",
    "high52_pct",
    "low52_pct",
    "kospi_relative_5d",
]

# ── 전역 momentum 모델 피처 — 52주 고점 모멘텀 (17개) ──────────────────────
FEATURE_COLS_MOMENTUM = [
    "high52_pct",         # 52주 고점 근접도 (핵심)
    "ma200_deviation",    # 장기 추세
    "mom_volatility",     # 20일 수익률 변동성
    "turnover_trend",     # 거래량 추세 (5일/20일)
    "volume_ratio",       # 거래량 비율
    "rsi",
    "bb_pct_20",
    "bb_std_20",
    "atr_pct",
    "ret_5d",
    "ret_10d",
    "kospi_relative_5d",
    "candle_body",
    "ema_deviation_20",
    "ma20_deviation",
    "ma_alignment",
    "ret_3d",
]

# ── 전역 midterm momentum 모델 피처 — 섹터 순환매 + 주도주 (8개) ────────────
FEATURE_COLS_MIDTERM_MOMENTUM = [
    "atr_pct",             # 가격 정규화 변동성
    "kospi_relative_20d",  # 20일 상대강도
    "beta_60d",            # 시장 베타
    "ma200_deviation",     # 장기 추세
    "ret_60d",             # 12주 모멘텀
    "ret_20d",             # 중기 모멘텀
    "high52_pct",          # 52주 고점 근접도
    "kospi_relative_5d",   # 5일 상대강도
]

# ── 전역 reversion 모델 피처 (12개) ─────────────────────────────────────────
# 제거: sector_relative_5d(12위), obv_change_5d(14위), volume_ratio(15위), volume_change(16위)
FEATURE_COLS_REVERSION = [
    "kospi_relative_5d",
    "low52_pct",
    "ret_5d",
    "bb_pct_20",
    "high52_pct",
    "ret_3d",
    "bb_std_20",
    "atr_pct",
    "rsi",
    "candle_body",
    "ema_deviation_20",
    "rsi_oversold",
]


def detect_momentum_rows(df: pd.DataFrame) -> pd.Series:
    """52주 고점 모멘텀 진입 후보 필터 (예측 시 사용, 학습에는 미적용).

    ① Close > MA200 (장기 상승 추세)
    ② high52_pct >= -0.10 (52주 고점 10% 이내)
    ③ 최근 3일 중 거래량 > 20일 평균 1일 이상
    """
    if "mom_volatility" not in df.columns:
        df = add_features(df)

    trend_up  = df["ma200_deviation"] > 0
    near_52w  = df["high52_pct"] >= -0.10

    vol_above   = (df["Volume"] > df["volume_ma20"]).astype(int)
    vol_recent3 = vol_above.rolling(3).sum() >= 1

    return (trend_up & near_52w & vol_recent3).fillna(False)


def detect_midterm_momentum_rows(df: pd.DataFrame) -> pd.Series:
    """중기 모멘텀 진입 후보 필터 (예측 시 사용, 학습에는 미적용).

    ① 섹터 5일 수익률 > KOSPI 5일 수익률 + 2% (sector_momentum_5d 있을 때만)
    ② Close > MA20
    ③ Close > MA200
    ④ Volume > 20일 평균 * 1.5
    """
    if "beta_60d" not in df.columns:
        df = add_features(df)

    above_ma20  = df["ma20_deviation"] > 0
    above_ma200 = df["ma200_deviation"] > 0
    vol_surge   = df["volume_ratio"] > 1.5

    if (df.get("sector_momentum_5d") is not None
            and df["sector_momentum_5d"].notna().any()):
        kospi_ret5   = df["ret_5d"] - df["kospi_relative_5d"].fillna(0)
        sector_ok    = df["sector_momentum_5d"] > (kospi_ret5 + 0.02)
    else:
        sector_ok = pd.Series(True, index=df.index)

    return (sector_ok & above_ma20 & above_ma200 & vol_surge).fillna(False)


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
