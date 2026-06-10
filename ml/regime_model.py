"""
ml/regime_model.py — 시장 레짐 예측 ML (방안 1)

타겟: 코스피 200 ETF(069500.KS) 향후 REGIME_HORIZON 거래일 수익률 >= 0 이면 1 (상승 레짐)
피처: 코스피 200의 MA 비율, 모멘텀, RSI, BB, 변동성 (10개)
모델: XGBoost (개별 종목 ML보다 학습 샘플 10× 이상 많아 과적합 위험 낮음)

walk-forward 백테스트에서 사용법:
    regime_model = train_regime(kospi_df, train_end_date)
    is_bull = predict_regime(regime_model, kospi_df, signal_date)  # True/False
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_HORIZON  = 20   # 향후 20거래일 수익률로 레짐 정의
REGIME_FEATURES = [
    "ma5_ratio", "ma20_ratio", "ma60_ratio", "ma200_ratio",
    "mom_10d", "mom_20d", "mom_60d",
    "vol_20d", "rsi_14", "bb_pct_20",
]


def _build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """코스피 200 일봉 → 레짐 예측 피처 생성."""
    close = df["Close"].squeeze().astype(float)
    f = pd.DataFrame(index=df.index)

    for w in (5, 20, 60, 200):
        ma = close.rolling(w).mean()
        f[f"ma{w}_ratio"] = close / ma.replace(0, np.nan) - 1

    for w in (10, 20, 60):
        f[f"mom_{w}d"] = close.pct_change(w)

    f["vol_20d"] = close.pct_change().rolling(20).std()

    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    avg_loss = (-delta).clip(lower=0).ewm(com=13, min_periods=14).mean()
    f["rsi_14"] = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))

    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    denom    = (bb_upper - bb_lower).replace(0, np.nan)
    f["bb_pct_20"] = (close - bb_lower) / denom

    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    return f


def train_regime(kospi_df: pd.DataFrame, train_end_date) -> object | None:
    """
    train_end_date 이전 코스피 200 데이터로 레짐 예측 모델 학습.

    반환: 학습된 XGBClassifier, 데이터 부족 시 None.
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("[Regime] xgboost 없음 — 레짐 필터 비활성화")
        return None

    df = kospi_df[kospi_df.index <= train_end_date].copy()
    if len(df) < 250:
        logger.debug("[Regime] 학습 데이터 부족 (%d행)", len(df))
        return None

    feat   = _build_regime_features(df)
    close  = df["Close"].squeeze().astype(float)
    future = close.shift(-REGIME_HORIZON) / close - 1
    target = (future >= 0).astype(int)

    combined = feat[REGIME_FEATURES].join(target.rename("_y")).dropna()
    # 마지막 REGIME_HORIZON행은 미래 레이블 없음 → 제거
    combined = combined.iloc[:-REGIME_HORIZON]

    X = combined[REGIME_FEATURES].values.astype("float32")
    y = combined["_y"].values.astype(int)

    if len(X) < 100:
        return None

    pos_r = y.mean()
    spw   = (1 - pos_r) / pos_r if 0 < pos_r < 1 else 1.0
    model = xgb.XGBClassifier(
        n_estimators      = 200,
        max_depth         = 3,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = spw,
        eval_metric       = "auc",
        random_state      = 42,
        verbosity         = 0,
    )
    model.fit(X, y)
    logger.debug("[Regime] 학습 완료: %d샘플, 상승레짐 비율=%.1f%%", len(X), pos_r * 100)
    return model


def predict_regime(model, kospi_df: pd.DataFrame, signal_date) -> bool:
    """
    signal_date 기준 코스피 레짐 예측.
    True=상승 레짐 (신호 허용), False=하락 레짐 (신호 차단).
    모델 없거나 데이터 부족이면 True (필터 비활성화 = 보수적으로 허용).
    """
    if model is None:
        return True

    hist = kospi_df[kospi_df.index <= signal_date]
    if len(hist) < 210:
        return True

    feat = _build_regime_features(hist)
    row  = feat[REGIME_FEATURES].iloc[[-1]].values.astype("float32")
    if np.isnan(row).any():
        return True

    prob = float(model.predict_proba(row)[0, 1])
    return prob >= 0.50
