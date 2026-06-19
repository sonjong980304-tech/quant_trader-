#!/usr/bin/env python3
"""
train_midterm_momentum.py — 중기 모멘텀 모델 학습 및 저장

피처 8개 (sector_momentum_5d 제거), TP=12% / SL=6% / 보유=20일
저장: ml/models/_global_midterm_momentum.pkl
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import csv, os, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score
import yfinance as yf

from ml.features import (
    add_features, _triple_barrier_pnl,
    detect_midterm_momentum_rows, FEATURE_COLS_MIDTERM_MOMENTUM,
)

# ── 파라미터 ──────────────────────────────────────────────────────────────
TP, SL, HOLD      = 0.12, 0.06, 20
BT_START, BT_END  = "2026-01-01", "2026-06-19"
WIN_PROB_THRESH   = 0.52
POSITION_SIZE     = 0.10
MAX_POSITIONS     = 10
COST_RT           = 0.0046
MODEL_PATH        = "ml/models/_global_midterm_momentum.pkl"

WF_FOLDS = [
    ("2023-01-01", "2024-01-01", "2025-01-01"),
    ("2023-01-01", "2025-01-01", "2026-01-01"),
    ("2023-01-01", "2026-01-01", "2027-01-01"),
]

FC = FEATURE_COLS_MIDTERM_MOMENTUM

# ── 과거 결과 (비교용) ────────────────────────────────────────────────────
BASELINES = {
    "14피처": {
        "oof_auc": 0.5285, "v2025": 0.5248, "v2026": 0.5527,
        "total_ret": 3.80, "sharpe": 1.363, "mdd": -2.99,
        "n_trades": 44, "win_rate": 50.0, "pf": 1.80,
        "wp_min": 0.2759, "wp_mean": 0.4627, "wp_max": 0.6652,
    },
    "9피처": {
        "oof_auc": 0.5288, "v2025": 0.5206, "v2026": 0.5631,
        "total_ret": 3.64, "sharpe": 0.794, "mdd": -5.17,
        "n_trades": 62, "win_rate": 43.5, "pf": 1.82,
        "wp_min": 0.2229, "wp_mean": 0.4661, "wp_max": 0.7377,
    },
    "reversion": {
        "oof_auc": 0.5674, "v2025": None, "v2026": None,
        "total_ret": 20.86, "sharpe": 1.976, "mdd": -7.16,
        "n_trades": 84, "win_rate": 59.5, "pf": 1.59,
        "wp_min": None, "wp_mean": None, "wp_max": None,
    },
}


def _strip_tz(idx):
    return idx.tz_localize(None) if idx.tzinfo else idx


def _to_ts(d):
    ts = pd.Timestamp(d)
    return ts.tz_localize(None) if ts.tzinfo else ts


def _row(df, date):
    idx = _strip_tz(df.index)
    pos = np.where(idx == date)[0]
    return df.iloc[pos] if len(pos) else pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# 1. 데이터 준비
# ════════════════════════════════════════════════════════════════════════════
print("KOSPI 다운로드...")
kospi_raw = yf.download("^KS11", period="3y", auto_adjust=True, progress=False)
if isinstance(kospi_raw.columns, pd.MultiIndex):
    kospi_raw.columns = kospi_raw.columns.get_level_values(0)
kospi_raw.index = _strip_tz(kospi_raw.index)

print("유니버스 로드...")
from signals.krx_universe import get_krx_backtest_universe
tickers = list(get_krx_backtest_universe(top_n=200).keys())

print(f"티커 데이터 다운로드 ({len(tickers)}개)...")
raw_data: dict[str, pd.DataFrame] = {}
for ticker in tickers:
    try:
        df = yf.download(ticker, period="3y", auto_adjust=True, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = _strip_tz(df.index)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        raw_data[ticker] = df
    except Exception:
        pass
print(f"다운로드 완료: {len(raw_data)}개")

print("피처 계산...")
feat_data: dict[str, pd.DataFrame] = {}
for ticker, df in raw_data.items():
    try:
        df_f = add_features(df, kospi_df=kospi_raw)
        df_f.index = _strip_tz(df_f.index)
        if len(df_f) >= 60:
            feat_data[ticker] = df_f
    except Exception:
        pass
print(f"피처 계산 완료: {len(feat_data)}개")

# ── 섹터 매핑 + sector_momentum_5d (detect 함수용) ───────────────────────
SECTOR_MAP_PATH = "ml/models/sector_map.csv"
sector_map: dict[str, str] = {}
if os.path.exists(SECTOR_MAP_PATH):
    with open(SECTOR_MAP_PATH, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) == 2:
                sector_map[row[0]] = row[1]
    print(f"섹터 매핑 로드: {len(sector_map)}개")

if sector_map:
    rows = []
    for ticker, df_f in feat_data.items():
        sector = sector_map.get(ticker, "unknown")
        sub = df_f[["ret_5d"]].copy()
        sub["_ticker"] = ticker
        sub["_sector"] = sector
        rows.append(sub)
    if rows:
        all_ret = pd.concat(rows).reset_index(names="_date")
        all_ret["_date"] = pd.to_datetime(all_ret["_date"])
        sect_avg = (
            all_ret[all_ret["_sector"] != "unknown"]
            .groupby(["_date", "_sector"])["ret_5d"].mean()
        )
        for ticker, df_f in feat_data.items():
            sector = sector_map.get(ticker, "unknown")
            if sector == "unknown": continue
            try:
                if isinstance(sect_avg.index, pd.MultiIndex):
                    sec_series = (sect_avg.xs(sector, level="_sector")
                                  if sector in sect_avg.index.get_level_values("_sector")
                                  else pd.Series(dtype=float))
                else:
                    sec_series = pd.Series(dtype=float)
                aligned = sec_series.reindex(pd.to_datetime(df_f.index).normalize()).values
                df_f["sector_momentum_5d"] = aligned
            except Exception:
                pass
    print("sector_momentum_5d 계산 완료 (detect 함수용)")

# ── FC dropna 후 학습용 clean 데이터 ─────────────────────────────────────
feat_clean: dict[str, pd.DataFrame] = {}
for ticker, df_f in feat_data.items():
    sub = df_f.dropna(subset=FC)
    if len(sub) >= 50:
        feat_clean[ticker] = sub
print(f"학습 가능 종목: {len(feat_clean)}개  |  피처 {len(FC)}개: {FC}")

# ════════════════════════════════════════════════════════════════════════════
# 2. 라벨 생성
# ════════════════════════════════════════════════════════════════════════════
print(f"\n라벨 생성 (TP={TP*100:.0f}%  SL={SL*100:.0f}%  hold={HOLD}d)...")
all_rows = []
for ticker, df_f in feat_clean.items():
    try:
        labels, future_ret = _triple_barrier_pnl(
            df_f, tp_pct=TP, sl_pct=SL, max_holding_days=HOLD
        )
        tmp = df_f.copy()
        tmp["_label"]  = labels
        tmp["_future"] = future_ret
        tmp["_ticker"] = ticker
        tmp["_date"]   = df_f.index
        tmp = tmp.dropna(subset=["_label","_future"]).iloc[:-HOLD]
        if len(tmp) >= 5:
            all_rows.append(tmp)
    except Exception:
        pass

combined = pd.concat(all_rows).sort_values("_date").reset_index(drop=True)
X     = combined[FC].values.astype("float32")
y     = combined["_label"].values.astype(int)
dates = pd.to_datetime(combined["_date"].values)
print(f"전체 샘플: {len(X):,}  |  양성 비율: {y.mean():.3f}")

# ════════════════════════════════════════════════════════════════════════════
# 3. Walk-forward CV
# ════════════════════════════════════════════════════════════════════════════
print("\nWalk-forward CV...")
oof_proba  = np.full(len(X), np.nan)
fold_aucs  = {}

for (tr_start, tr_end, vl_end) in WF_FOLDS:
    tr_mask = (dates >= tr_start) & (dates < tr_end)
    vl_mask = (dates >= tr_end)   & (dates < vl_end)
    if tr_mask.sum() < 20 or vl_mask.sum() < 20:
        continue
    tr_idx = np.where(tr_mask)[0]
    vl_idx = np.where(vl_mask)[0]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_vl, y_vl = X[vl_idx], y[vl_idx]

    pos_r = y_tr.mean()
    spw   = (1 - pos_r) / pos_r if pos_r > 0 else 1.0
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc", random_state=42, verbosity=0,
    )
    m.fit(X_tr, y_tr)
    proba = m.predict_proba(X_vl)[:, 1]
    oof_proba[vl_idx] = proba
    try:
        fa    = roc_auc_score(y_vl, proba)
        lbl   = f"v{pd.Timestamp(tr_end).year}"
        fold_aucs[lbl] = round(fa, 4)
        print(f"  {lbl}: AUC={fa:.4f}  (n={vl_mask.sum():,})")
    except Exception:
        pass

valid_mask = ~np.isnan(oof_proba)
oof_auc    = roc_auc_score(y[valid_mask], oof_proba[valid_mask]) if valid_mask.any() else 0.0
print(f"\nOOF AUC: {oof_auc:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 4. 최종 모델 학습 + 저장
# ════════════════════════════════════════════════════════════════════════════
print("\n최종 모델 학습...")
n            = len(X)
calib_size   = max(20, int(n * 0.20))
X_tr2, y_tr2 = X[:-calib_size], y[:-calib_size]
X_cal, y_cal = X[-calib_size:], y[-calib_size:]

pos_r = y_tr2.mean()
spw   = (1 - pos_r) / pos_r if pos_r > 0 else 1.0
final = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw,
    eval_metric="auc", random_state=42, verbosity=0,
)
final.fit(X_tr2, y_tr2)
cal_model = CalibratedClassifierCV(FrozenEstimator(final), method="sigmoid")
cal_model.fit(X_cal, y_cal)

os.makedirs("ml/models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "model":        cal_model,
        "feature_cols": FC,
        "oof_auc":      round(oof_auc, 4),
        "tp": TP, "sl": SL, "hold": HOLD,
        "n_samples":    n,
    }, f)
print(f"모델 저장: {MODEL_PATH}")

# ── 피처 중요도 ───────────────────────────────────────────────────────────
fi = sorted(zip(FC, final.feature_importances_), key=lambda x: -x[1])

shap_fi = None
try:
    import shap
    exp       = shap.TreeExplainer(final)
    sv        = exp.shap_values(X[:min(5000, len(X))])
    shap_mean = np.abs(sv).mean(axis=0)
    shap_fi   = sorted(zip(FC, shap_mean.tolist()), key=lambda x: -x[1])
except Exception:
    pass

# ── win_prob 분포 ─────────────────────────────────────────────────────────
op_cal  = cal_model.predict_proba(X[valid_mask])[:, 1]
wp_min  = round(float(op_cal.min()),  4)
wp_mean = round(float(op_cal.mean()), 4)
wp_max  = round(float(op_cal.max()),  4)
above   = {t: round(float((op_cal >= t).mean()), 4) for t in [0.50, 0.52, 0.55]}

# ════════════════════════════════════════════════════════════════════════════
# 5. 백테스트
# ════════════════════════════════════════════════════════════════════════════
print(f"\n백테스트 ({BT_START} ~ {BT_END})...")
all_days = _strip_tz(kospi_raw.index)
bt_days  = all_days[(all_days >= BT_START) & (all_days <= BT_END)]

signals_by_date: dict[pd.Timestamp, list[dict]] = {}
for ticker, df_f in feat_clean.items():
    try:
        sig_range = df_f[(df_f.index >= BT_START) & (df_f.index < BT_END)]
        mask      = detect_midterm_momentum_rows(sig_range).reindex(sig_range.index).fillna(False)
        for sig_date in sig_range[mask].index:
            row = sig_range.loc[sig_date]
            Xp  = row[FC].values.reshape(1, -1).astype("float32")
            wp  = float(cal_model.predict_proba(Xp)[0, 1])
            if wp >= WIN_PROB_THRESH:
                ts = _to_ts(sig_date)
                signals_by_date.setdefault(ts, []).append(
                    {"ticker": ticker, "win_prob": wp}
                )
    except Exception:
        pass

capital   = 1.0
positions: list[dict] = []
trades:    list[dict] = []
daily_nav: list[float] = []

for i, today in enumerate(bt_days):
    next_day = bt_days[i + 1] if i + 1 < len(bt_days) else None

    still_open = []
    for pos in positions:
        pos["hold_days"] += 1
        df_pos    = raw_data.get(pos["ticker"])
        today_row = _row(df_pos, today) if df_pos is not None else pd.DataFrame()
        if today_row.empty:
            still_open.append(pos); continue
        hi, lo, op_px = (float(today_row[c].iloc[0]) for c in ("High","Low","Open"))
        ep = pos["entry_price"]
        exit_px = exit_rsn = None
        if lo <= ep * (1 - SL):
            exit_px, exit_rsn = ep * (1 - SL), "SL"
        elif hi >= ep * (1 + TP):
            exit_px, exit_rsn = ep * (1 + TP), "TP"
        elif pos["hold_days"] >= HOLD:
            exit_px, exit_rsn = op_px, "TIME"
        if exit_px is not None:
            net = exit_px / ep - 1 - COST_RT
            capital += pos["size"] * (1 + net)
            trades.append({"net_ret": net, "exit_reason": exit_rsn, "win": net > 0})
        else:
            still_open.append(pos)
    positions = still_open

    if next_day is not None and today in signals_by_date:
        cands = sorted(signals_by_date[today], key=lambda x: -x["win_prob"])
        for sig in cands:
            if len(positions) >= MAX_POSITIONS: break
            if any(p["ticker"] == sig["ticker"] for p in positions): continue
            df_pos = raw_data.get(sig["ticker"])
            if df_pos is None: continue
            entry_row = _row(df_pos, next_day)
            if entry_row.empty: continue
            ep = float(entry_row["Open"].iloc[0])
            if ep <= 0: continue
            size = capital * POSITION_SIZE
            capital -= size * (1 + COST_RT / 2)
            positions.append({"ticker": sig["ticker"], "entry_price": ep,
                               "hold_days": 0, "size": size})

    pos_mkt = 0.0
    for pos in positions:
        df_pos = raw_data.get(pos["ticker"])
        tr     = _row(df_pos, today) if df_pos is not None else pd.DataFrame()
        cp     = float(tr["Close"].iloc[0]) if not tr.empty else pos["entry_price"]
        pos_mkt += pos["size"] * (cp / pos["entry_price"])
    daily_nav.append(capital + pos_mkt)

# ── 백테스트 지표 계산 ────────────────────────────────────────────────────
if trades and len(daily_nav) >= 2:
    df_tr  = pd.DataFrame(trades)
    nav    = pd.Series(daily_nav)
    dr     = nav.pct_change().dropna()
    wins   = df_tr[df_tr["net_ret"] > 0]["net_ret"]
    losses = df_tr[df_tr["net_ret"] <= 0]["net_ret"]
    bt = {
        "total_ret": round((nav.iloc[-1] - 1) * 100, 2),
        "sharpe":    round(dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0, 3),
        "mdd":       round(((nav - nav.cummax()) / nav.cummax()).min() * 100, 2),
        "n_trades":  len(df_tr),
        "win_rate":  round(df_tr["win"].mean() * 100, 1),
        "avg_win":   round(wins.mean() * 100, 2) if len(wins) > 0 else 0.0,
        "avg_loss":  round(losses.mean() * 100, 2) if len(losses) > 0 else 0.0,
        "pf":        round(abs(wins.mean() / losses.mean()), 2)
                     if len(losses) > 0 and losses.mean() != 0 else 0.0,
        "exit":      df_tr["exit_reason"].value_counts().to_dict(),
    }
else:
    bt = None

# ════════════════════════════════════════════════════════════════════════════
# 6. 결과 출력 (3-way 비교)
# ════════════════════════════════════════════════════════════════════════════
B14 = BASELINES["14피처"]
B9  = BASELINES["9피처"]
R   = BASELINES["reversion"]

def _fmt(v, fmt=".2f", suffix=""):
    return f"{v:{fmt}}{suffix}" if v is not None else "—"

W = 14  # 열 너비

print(f"\n{'='*72}")
print(f"중기 모멘텀 피처 축소 비교  (TP=12% / SL=6% / hold=20d)")
print(f"{'='*72}")
print(f"{'':>22}  {'14피처':>{W}}  {'9피처':>{W}}  {'8피처(이번)':>{W}}")
print("-" * 72)

# OOF AUC
v14 = _fmt(B14["oof_auc"], ".4f")
v9  = _fmt(B9["oof_auc"],  ".4f")
v8  = _fmt(oof_auc,         ".4f")
print(f"{'OOF AUC':>22}  {v14:>{W}}  {v9:>{W}}  {v8:>{W}}")

v14 = _fmt(B14["v2025"], ".4f")
v9  = _fmt(B9["v2025"],  ".4f")
v8  = _fmt(fold_aucs.get("v2025"), ".4f") if fold_aucs.get("v2025") else "—"
print(f"{'  valid 2025':>22}  {v14:>{W}}  {v9:>{W}}  {v8:>{W}}")

v14 = _fmt(B14["v2026"], ".4f")
v9  = _fmt(B9["v2026"],  ".4f")
v8  = _fmt(fold_aucs.get("v2026"), ".4f") if fold_aucs.get("v2026") else "—"
print(f"{'  valid 2026':>22}  {v14:>{W}}  {v9:>{W}}  {v8:>{W}}")

print("-" * 72)

# 백테스트
rows_bt = [
    ("수익률",   "total_ret", ".2f", "%"),
    ("샤프",     "sharpe",    ".3f", ""),
    ("MDD",      "mdd",       ".2f", "%"),
    ("거래 수",  "n_trades",  "d",   "건"),
    ("승률",     "win_rate",  ".1f", "%"),
    ("손익비",   "pf",        ".2f", ""),
]
for label, key, fmt, suffix in rows_bt:
    v14 = f"{B14[key]:{fmt}}{suffix}"
    v9  = f"{B9[key]:{fmt}}{suffix}"
    v8  = f"{bt[key]:{fmt}}{suffix}" if bt else "—"
    print(f"{label:>22}  {v14:>{W}}  {v9:>{W}}  {v8:>{W}}")

print("-" * 72)

# win_prob 분포
print(f"{'win_prob min':>22}  {_fmt(B14['wp_min'],'.4f'):>{W}}  {_fmt(B9['wp_min'],'.4f'):>{W}}  {_fmt(wp_min,'.4f'):>{W}}")
print(f"{'win_prob mean':>22}  {_fmt(B14['wp_mean'],'.4f'):>{W}}  {_fmt(B9['wp_mean'],'.4f'):>{W}}  {_fmt(wp_mean,'.4f'):>{W}}")
print(f"{'win_prob max':>22}  {_fmt(B14['wp_max'],'.4f'):>{W}}  {_fmt(B9['wp_max'],'.4f'):>{W}}  {_fmt(wp_max,'.4f'):>{W}}")
print(f"{'>=0.52 비율':>22}  {'12.9%':>{W}}  {'18.7%':>{W}}  {above[0.52]*100:.1f}%{'':{W-len(f'{above[0.52]*100:.1f}%')}}")

print(f"\n{'='*72}")
print(f"reversion 비교")
print(f"{'='*72}")
print(f"{'':>22}  {'reversion':>{W}}  {'8피처(이번)':>{W}}")
print("-" * 50)
for label, key, fmt, suffix in rows_bt:
    vr = f"{R[key]:{fmt}}{suffix}"
    v8 = f"{bt[key]:{fmt}}{suffix}" if bt else "—"
    print(f"{label:>22}  {vr:>{W}}  {v8:>{W}}")

# 피처 중요도
print(f"\n{'='*72}")
print(f"피처 중요도 (8피처)")
print(f"{'='*72}")
fi_to_show = shap_fi or fi
fi_label   = "SHAP" if shap_fi else "Feature Importance"
print(f"  ({fi_label})")
for rank, (feat, imp) in enumerate(fi_to_show, 1):
    print(f"  {rank}. {feat:25s}  {imp:.4f}")

if bt:
    print(f"\n  청산 사유: {bt['exit']}")
