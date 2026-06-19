from __future__ import annotations

"""
model.py - XGBoost 7일 수익 예측 모델

학습: TimeSeriesSplit(5-fold) → 전체 데이터로 최종 모델 학습
예측: 최신 바의 피처로 승률(win_prob) 반환
저장: ml/models/{ticker}.pkl
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from ml.features import add_features, _triple_barrier_pnl, FEATURE_COLS
from config import TP_PCT, SL_PCT, EOD_HORIZON

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

HORIZON    = EOD_HORIZON   # 최대 보유 거래일 수 (config.EOD_HORIZON)
N_SPLITS   = 5   # TimeSeriesSplit fold 수


def _model_path(ticker: str, agent: str = "") -> str:
    suffix = f"_{agent}" if agent else ""
    return os.path.join(MODEL_DIR, f"{ticker.replace('.', '_')}{suffix}.pkl")


def train(df: pd.DataFrame, ticker: str, agent: str = "") -> tuple[object, dict]:
    """
    XGBoost 모델 학습 후 pkl 저장.

    df     : 5~10년치 OHLCV 데이터프레임
    ticker : 종목 티커 (파일명 키)
    agent  : "" (전체) | "momentum" (돌파) | "reversion" (눌림목)

    반환: (model, metrics)
    metrics 키: accuracy, auc, avg_win, avg_loss, n_samples, positive_rate
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost가 설치되어 있지 않습니다. pip install xgboost")

    df = add_features(df)
    labels, future_returns = _triple_barrier_pnl(
        df, tp_pct=TP_PCT, sl_pct=SL_PCT, max_holding_days=HORIZON
    )

    df = df.copy()
    df["_label"]         = labels
    df["_future_return"] = future_returns
    df = df.dropna(subset=FEATURE_COLS + ["_label", "_future_return"])
    df = df.iloc[:-HORIZON]   # 미래 데이터 부족한 마지막 N행 제거

    # 에이전트별 트리거 조건 필터링
    if agent == "momentum":
        from ml.features import detect_momentum_rows
        mask = detect_momentum_rows(df).reindex(df.index).fillna(False)
        df = df[mask]
        logger.info("  [%s] momentum 필터 후 %d행", ticker, len(df))
    elif agent == "reversion":
        from ml.features import detect_reversion_rows
        mask = detect_reversion_rows(df).reindex(df.index).fillna(False)
        df = df[mask]
        logger.info("  [%s] reversion 필터 후 %d행", ticker, len(df))

    X            = df[FEATURE_COLS].values.astype(np.float32)
    y            = df["_label"].values.astype(int)
    future_ret   = df["_future_return"].values

    min_samples = 50 if agent else 100
    if len(X) < min_samples:
        raise ValueError(f"학습 데이터 부족: {len(X)}행 (최소 {min_samples}행 필요)")

    tscv         = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_preds    = np.zeros(len(X), dtype=int)
    oof_proba    = np.zeros(len(X), dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        pos_ratio        = y_tr.mean()
        scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0

        fold_model = xgb.XGBClassifier(
            n_estimators      = 300,
            max_depth         = 4,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            scale_pos_weight  = scale_pos_weight,
            eval_metric       = "auc",
            random_state      = 42,
            verbosity         = 0,
        )
        fold_model.fit(
            X_tr, y_tr,
            eval_set   = [(X_val, y_val)],
            verbose    = False,
        )
        oof_preds[val_idx] = fold_model.predict(X_val)
        oof_proba[val_idx] = fold_model.predict_proba(X_val)[:, 1]
        logger.info("  [%s] Fold %d acc=%.3f", ticker, fold + 1,
                    accuracy_score(y_val, oof_preds[val_idx]))

    # 마지막 20%를 캘리브레이션 홀드아웃으로 분리 (시계열 순서 유지, 누수 방지)
    # final_model은 홀드아웃을 학습에 포함하지 않으므로 진정한 OOS 캘리브레이션 보장
    n          = len(X)
    calib_size = max(20, int(n * 0.20))
    if calib_size >= n - 50:          # 학습 데이터가 너무 적어지면 축소
        calib_size = max(20, n // 5)

    X_final_tr = X[:-calib_size]
    y_final_tr = y[:-calib_size]
    X_calib    = X[-calib_size:]
    y_calib    = y[-calib_size:]

    pos_ratio        = y_final_tr.mean() if len(y_final_tr) > 0 else y.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0
    final_model = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos_weight,
        eval_metric       = "auc",
        random_state      = 42,
        verbosity         = 0,
    )
    final_model.fit(X_final_tr, y_final_tr)

    # Platt Scaling 미적용 — raw XGBoost 확률 사용
    calibrated_model = final_model
    brier_raw = np.nan
    brier_cal = np.nan
    if len(X_calib) >= 20:
        proba_raw = final_model.predict_proba(X_calib)[:, 1]
        brier_raw = float(brier_score_loss(y_calib, proba_raw))
        brier_cal = brier_raw

    # OOF 메트릭 — TimeSeriesSplit 첫 구간은 validation에 포함되지 않아
    # oof_proba가 초기값 0.0으로 남으므로 실제 예측된 인덱스만 사용
    valid_mask = oof_proba > 0
    acc = accuracy_score(y[valid_mask], oof_preds[valid_mask]) if valid_mask.any() else 0.0
    try:
        auc = roc_auc_score(y[valid_mask], oof_proba[valid_mask]) if valid_mask.any() else 0.0
    except Exception:
        auc = 0.0

    wins   = future_ret[y == 1]
    losses = future_ret[y == 0]
    avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0

    metrics = {
        "accuracy":      round(acc, 4),
        "auc":           round(auc, 4),
        "avg_win":       round(avg_win, 4),
        "avg_loss":      round(avg_loss, 4),
        "n_samples":     len(X),
        "positive_rate": round(float(y.mean()), 4),
        "tp_pct":        TP_PCT,
        "sl_pct":        SL_PCT,
        "horizon":       HORIZON,
        "brier_raw":     round(brier_raw, 4) if not np.isnan(brier_raw) else None,
        "brier_cal":     round(brier_cal, 4) if not np.isnan(brier_cal) else None,
        "feature_cols":  FEATURE_COLS,
    }

    path = _model_path(ticker, agent)
    with open(path, "wb") as f:
        pickle.dump({"model": calibrated_model, "metrics": metrics}, f)

    agent_label  = f"[{agent}] " if agent else ""
    brier_log    = (f" brier_raw={brier_raw:.4f}→cal={brier_cal:.4f}"
                    if not np.isnan(brier_raw) else "")
    logger.info("[%s] %s모델 저장 완료 | acc=%.3f auc=%.3f avg_win=%.1f%% avg_loss=%.1f%%%s",
                ticker, agent_label, acc, auc, avg_win * 100, avg_loss * 100, brier_log)
    return final_model, metrics


def load_model(ticker: str, agent: str = "") -> tuple[object | None, dict | None]:
    """저장된 모델 로드. 없으면 (None, None) 반환."""
    path = _model_path(ticker, agent)
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["metrics"]


# ── KOSPI 캐시 (예측 시 kospi_relative_5d 계산용) ─────────────────────────────
_KOSPI_CACHE: dict = {}

def _get_kospi_df() -> pd.DataFrame | None:
    """KOSPI 일봉 데이터 일별 캐시. 실패 시 None."""
    from datetime import datetime as _dt
    cache_key = _dt.now().strftime("%Y-%m-%d")
    if _KOSPI_CACHE.get("key") == cache_key:
        return _KOSPI_CACHE.get("df")
    try:
        import yfinance as yf
        df = yf.download("^KS11", period="1y", auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        _KOSPI_CACHE.update({"key": cache_key, "df": df})
        return df
    except Exception as e:
        logger.debug("[KOSPI 캐시] 조회 실패: %s", e)
        return None


_DEFAULT_WF_FOLDS = [
    ("2020-01-01", "2022-01-01", "2023-01-01"),
    ("2020-01-01", "2023-01-01", "2024-01-01"),
    ("2020-01-01", "2024-01-01", "2025-01-01"),
    ("2020-01-01", "2025-01-01", "2027-01-01"),
]


def train_global(combined_df: pd.DataFrame, agent: str,
                 feature_cols: list[str] | None = None,
                 wf_folds: list[tuple] | None = None) -> tuple[object, dict]:
    """
    전체 종목 합산 DataFrame으로 단일 전역 모델 학습.

    feature_cols: 사용할 피처 목록. None이면 FEATURE_COLS 사용.
    wf_folds: walk-forward split 리스트. None이면 _DEFAULT_WF_FOLDS 사용.
              각 원소: (train_start, train_end, val_end)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    if wf_folds is None:
        wf_folds = _DEFAULT_WF_FOLDS

    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost가 설치되어 있지 않습니다. pip install xgboost")

    combined_df = combined_df.copy()
    X          = combined_df[feature_cols].values.astype(np.float32)
    y          = combined_df["_label"].values.astype(int)
    future_ret = combined_df["_future_return"].values
    dates      = pd.to_datetime(combined_df["_date"]).dt.tz_localize(None)

    oof_proba  = np.full(len(X), np.nan)
    fold_aucs  = []

    for train_start, train_end, val_end in wf_folds:
        val_start = train_end
        tr_mask = (dates >= pd.Timestamp(train_start)) & (dates < pd.Timestamp(train_end))
        vl_mask = (dates >= pd.Timestamp(val_start))   & (dates < pd.Timestamp(val_end))

        tr_idx = np.where(tr_mask.values)[0]
        vl_idx = np.where(vl_mask.values)[0]

        if len(tr_idx) < 50 or len(vl_idx) < 10:
            logger.warning("[global/%s] fold 데이터 부족 train=%d val=%d — 건너뜀",
                           agent, len(tr_idx), len(vl_idx))
            continue

        X_tr, X_val = X[tr_idx], X[vl_idx]
        y_tr, y_val = y[tr_idx], y[vl_idx]

        pos_ratio        = y_tr.mean()
        scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0

        fold_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc", random_state=42, verbosity=0,
        )
        fold_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_proba[vl_idx] = fold_model.predict_proba(X_val)[:, 1]

        fold_acc = accuracy_score(y_val, fold_model.predict(X_val))
        try:
            fold_auc = roc_auc_score(y_val, fold_model.predict_proba(X_val)[:, 1])
        except Exception:
            fold_auc = 0.0
        fold_label = f"valid {val_start[:4]}"
        fold_aucs.append((fold_label, round(fold_auc, 4)))
        logger.info("[global/%s] train=%d val=%d acc=%.3f auc=%.4f",
                    agent, len(tr_idx), len(vl_idx), fold_acc, fold_auc)

    # OOF 메트릭 (validation에 포함된 행만)
    valid_mask = ~np.isnan(oof_proba)
    try:
        auc = roc_auc_score(y[valid_mask], oof_proba[valid_mask]) if valid_mask.any() else 0.0
    except Exception:
        auc = 0.0
    oof_preds = np.where(valid_mask, (oof_proba >= 0.5).astype(int), 0)
    acc = accuracy_score(y[valid_mask], oof_preds[valid_mask]) if valid_mask.any() else 0.0

    # 최종 모델 (전체 데이터)
    n          = len(X)
    calib_size = max(20, int(n * 0.20))
    X_final_tr, y_final_tr = X[:-calib_size], y[:-calib_size]
    X_calib,    y_calib    = X[-calib_size:], y[-calib_size:]

    pos_ratio_tr     = y_final_tr.mean() if len(y_final_tr) > 0 else y.mean()
    scale_pos_weight = (1 - pos_ratio_tr) / pos_ratio_tr if pos_ratio_tr > 0 else 1.0

    final_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc", random_state=42, verbosity=0,
    )
    final_model.fit(X_final_tr, y_final_tr)

    # Platt Scaling 미적용 — raw XGBoost 확률 사용
    calibrated_model = final_model
    brier_raw = brier_cal = np.nan
    if len(X_calib) >= 20:
        proba_raw = final_model.predict_proba(X_calib)[:, 1]
        brier_raw = float(brier_score_loss(y_calib, proba_raw))
        brier_cal = brier_raw

    wins   = future_ret[y == 1]
    losses = future_ret[y == 0]
    avg_win  = float(wins.mean())        if len(wins)   > 0 else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0

    fi_pairs = sorted(zip(feature_cols, final_model.feature_importances_),
                      key=lambda x: x[1], reverse=True)

    _op = oof_proba[valid_mask]
    oof_proba_stats = {
        "min":      round(float(_op.min()),  4) if valid_mask.any() else None,
        "max":      round(float(_op.max()),  4) if valid_mask.any() else None,
        "mean":     round(float(_op.mean()), 4) if valid_mask.any() else None,
        "p50":      round(float(np.percentile(_op, 50)), 4) if valid_mask.any() else None,
        "p75":      round(float(np.percentile(_op, 75)), 4) if valid_mask.any() else None,
        "p90":      round(float(np.percentile(_op, 90)), 4) if valid_mask.any() else None,
        "above_50": round(float((_op >= 0.50).mean()), 4) if valid_mask.any() else None,
        "above_52": round(float((_op >= 0.52).mean()), 4) if valid_mask.any() else None,
        "above_55": round(float((_op >= 0.55).mean()), 4) if valid_mask.any() else None,
    }

    metrics = {
        "accuracy":                  round(acc, 4),
        "auc":                       round(auc, 4),
        "avg_win":                   round(avg_win, 4),
        "avg_loss":                  round(avg_loss, 4),
        "n_samples":                 len(X),
        "positive_rate":             round(float(y.mean()), 4),
        "tp_pct":                    TP_PCT,
        "sl_pct":                    SL_PCT,
        "horizon":                   HORIZON,
        "brier_raw":                 round(brier_raw, 4) if not np.isnan(brier_raw) else None,
        "brier_cal":                 round(brier_cal, 4) if not np.isnan(brier_cal) else None,
        "feature_cols":              feature_cols,
        "fold_aucs":                 fold_aucs,
        "feature_importance_top10":  fi_pairs[:10],
        "feature_importance_all":    fi_pairs,
        "oof_proba_stats":           oof_proba_stats,
    }

    path = _model_path("_global", agent)
    with open(path, "wb") as f:
        pickle.dump({"model": calibrated_model, "metrics": metrics}, f)

    brier_log = (f" brier={brier_raw:.4f}→{brier_cal:.4f}"
                 if not np.isnan(brier_raw) else "")
    logger.info("[global/%s] 저장 완료 n=%d pos=%.3f acc=%.3f auc=%.4f%s",
                agent, len(X), y.mean(), acc, auc, brier_log)
    return calibrated_model, metrics


def predict(df: pd.DataFrame, ticker: str, agent: str = "") -> dict:
    """
    최신 바의 피처로 7일 승률 예측.
    전역 모델(_global)을 우선 사용하고, 없으면 종목별 모델로 폴백.
    """
    # 전역 모델 우선, 없으면 종목별 폴백
    model, metrics = load_model("_global", agent)
    if model is None:
        model, metrics = load_model(ticker, agent)
    if model is None:
        return {"has_model": False, "win_prob": None,
                "avg_win": None, "avg_loss": None}

    feature_cols = metrics.get("feature_cols", FEATURE_COLS)

    kospi_df = _get_kospi_df()
    df_feat  = add_features(df, kospi_df=kospi_df)

    if "kospi_relative_5d" in df_feat.columns:
        df_feat["kospi_relative_5d"] = df_feat["kospi_relative_5d"].fillna(0.0)

    # sector_relative_5d: 예측 시 단일 종목만 있으므로 kospi_relative_5d 로 대체
    if "sector_relative_5d" in feature_cols and "sector_relative_5d" not in df_feat.columns:
        df_feat["sector_relative_5d"] = df_feat.get(
            "kospi_relative_5d", pd.Series(0.0, index=df_feat.index)
        )

    df_feat = df_feat.dropna(subset=feature_cols)
    if df_feat.empty:
        return {"has_model": True, "win_prob": None,
                "avg_win": metrics["avg_win"], "avg_loss": metrics["avg_loss"]}

    X_latest = df_feat[feature_cols].iloc[[-1]].values.astype(np.float32)
    win_prob = float(model.predict_proba(X_latest)[0, 1])

    return {
        "has_model":  True,
        "win_prob":   round(win_prob, 4),
        "avg_win":    metrics["avg_win"],
        "avg_loss":   metrics["avg_loss"],
        "model_acc":  metrics["accuracy"],
        "model_auc":  metrics["auc"],
    }
