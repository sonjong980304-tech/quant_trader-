#!/usr/bin/env python3
"""
combined_backtest.py — reversion + trend 합산 워크포워드 백테스트

WF 정합:
  2024: reversion Fold1 (train 2023)     + trend rules
  2025: reversion Fold2 (train 2023-24)  + trend rules
  2026: reversion Fold3 (train 2023-25)  + trend rules
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from ml.features import (
    add_features, _triple_barrier_pnl,
    detect_reversion_rows, FEATURE_COLS_REVERSION,
)

# ═══════════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════════
BT_START, BT_END = "2024-01-01", "2026-06-19"

REV_TP, REV_SL, REV_HOLD = 0.15, 0.08, 10
REV_WP = 0.52
REV_PF = REV_TP / REV_SL   # 1.875

TR_ADX, TR_TRAIL, TR_VOL = 25, 2.0, 1.3
TR_HOLD  = 60
ADX_EXIT = 20

MAX_POS   = 10
MAX_ALLOC = 0.20
ATR_RISK  = 0.01
COST_RT   = 0.0046

WF_FOLDS = [
    ("2023-01-01", "2024-01-01", "2024"),
    ("2023-01-01", "2025-01-01", "2025"),
    ("2023-01-01", "2026-01-01", "2026"),
]

FC_REV = FEATURE_COLS_REVERSION


# ═══════════════════════════════════════════════════════════════════
# 유틸
# ═══════════════════════════════════════════════════════════════════
def _strip_tz(idx): return idx.tz_localize(None) if idx.tzinfo else idx
def _to_ts(d):
    ts = pd.Timestamp(d)
    return ts.tz_localize(None) if ts.tzinfo else ts

def _row(df, date):
    if df is None: return pd.DataFrame()
    idx = _strip_tz(df.index)
    p = np.where(idx == date)[0]
    return df.iloc[p] if len(p) else pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    for p in [5, 20, 60, 200]:
        df[f"ma{p}"] = c.rolling(p).mean()
    tr = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    df["tr"]      = tr
    df["atr"]     = tr.ewm(com=13, adjust=False).mean()
    df["vol_ma20"] = v.rolling(20).mean()
    dm_p = (h - h.shift(1)).clip(lower=0)
    dm_m = (l.shift(1) - l).clip(lower=0)
    dm_p = dm_p.where(dm_p >= dm_m, 0.0)
    dm_m = dm_m.where(dm_m >  dm_p, 0.0)
    atr_w  = tr.ewm(com=13, adjust=False).mean()
    di_p   = 100 * dm_p.ewm(com=13, adjust=False).mean() / atr_w
    di_m   = 100 * dm_m.ewm(com=13, adjust=False).mean() / atr_w
    dx     = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    df["adx"]    = dx.ewm(com=13, adjust=False).mean()
    df["Volume"] = v
    return df

def calc_rev_size(capital, win_prob, atr, ep):
    f = max(0.0, (win_prob * REV_PF - (1 - win_prob)) / REV_PF) * 0.5 * capital
    a = (capital * ATR_RISK * ep) / (2 * atr) if atr > 0 and ep > 0 else f
    return max(0.0, min(f, a, capital * MAX_ALLOC))

def calc_tr_size(capital, atr, ep):
    a = (capital * ATR_RISK * ep) / (2 * atr) if atr > 0 and ep > 0 else capital * 0.10
    return max(0.0, min(a, capital * MAX_ALLOC))


# ═══════════════════════════════════════════════════════════════════
# 1. 데이터 다운로드
# ═══════════════════════════════════════════════════════════════════
print("KOSPI 다운로드 (4년치)...")
kospi_raw = yf.download("^KS11", period="4y", auto_adjust=True, progress=False)
if isinstance(kospi_raw.columns, pd.MultiIndex):
    kospi_raw.columns = kospi_raw.columns.get_level_values(0)
kospi_raw.index  = _strip_tz(kospi_raw.index)
kospi_ind        = compute_indicators(kospi_raw)
kospi_ma200      = kospi_ind["ma200"].dropna()
kospi_close      = kospi_raw["Close"]

print("유니버스 로드...")
from signals.krx_universe import get_krx_backtest_universe
tickers = list(get_krx_backtest_universe(top_n=200).keys())

print(f"티커 다운로드 ({len(tickers)}개)...")
raw_data: dict[str, pd.DataFrame] = {}
for t in tickers:
    try:
        df = yf.download(t, period="4y", auto_adjust=True, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = _strip_tz(df.index)
        df = df[["Open","High","Low","Close","Volume"]].dropna()
        raw_data[t] = df
    except Exception: pass
print(f"완료: {len(raw_data)}개")

print("피처 + 지표 계산...")
feat_data: dict[str, pd.DataFrame] = {}
ind_data:  dict[str, pd.DataFrame] = {}
for t, df in raw_data.items():
    try:
        df_f = add_features(df, kospi_df=kospi_raw)
        df_f.index = _strip_tz(df_f.index)
        if len(df_f.dropna(subset=FC_REV)) >= 50:
            feat_data[t] = df_f
    except Exception: pass
    try:
        df_i = compute_indicators(df)
        if len(df_i.dropna(subset=["ma200","adx"])) >= 20:
            ind_data[t] = df_i
    except Exception: pass
print(f"리버전 피처: {len(feat_data)}개  |  트렌드 지표: {len(ind_data)}개")


# ═══════════════════════════════════════════════════════════════════
# 2. 리버전 WF 학습 데이터 구축
# ═══════════════════════════════════════════════════════════════════
print("\n리버전 WF 학습 데이터 구축...")
rev_rows = []
for t, df_f in feat_data.items():
    try:
        labels_arr, _ = _triple_barrier_pnl(
            df_f, tp_pct=REV_TP, sl_pct=REV_SL, max_holding_days=REV_HOLD
        )
        labels_s  = pd.Series(labels_arr, index=df_f.index)
        det_mask  = detect_reversion_rows(df_f).reindex(df_f.index, fill_value=False)
        sub       = df_f.dropna(subset=FC_REV)
        det_sub   = det_mask.reindex(sub.index, fill_value=False)
        lbl_sub   = labels_s.reindex(sub.index)
        valid     = det_sub & lbl_sub.notna()
        sub2      = sub[valid].copy()
        if len(sub2) < 5: continue
        sub2["_label"]  = lbl_sub[valid].values
        sub2["_ticker"] = t
        sub2["_date"]   = sub2.index
        rev_rows.append(sub2)
    except Exception: pass

rev_df = pd.concat(rev_rows).sort_values("_date").reset_index(drop=True)
X_all  = rev_df[FC_REV].values.astype("float32")
y_all  = rev_df["_label"].values.astype(int)
d_all  = pd.to_datetime(rev_df["_date"].values)
print(f"전체 리버전 샘플: {len(X_all):,}  |  양성 비율: {y_all.mean():.3f}")


# ═══════════════════════════════════════════════════════════════════
# 3. WF 폴드 모델 학습 (3개)
# ═══════════════════════════════════════════════════════════════════
print("\nWF 폴드 모델 학습...")
fold_models: dict[str, object] = {}

for (tr_s, tr_e, ylbl) in WF_FOLDS:
    mask = (d_all >= tr_s) & (d_all < tr_e)
    if mask.sum() < 50:
        print(f"  Fold {ylbl}: 샘플 부족 ({mask.sum()})")
        fold_models[ylbl] = None
        continue
    X_tr, y_tr = X_all[mask], y_all[mask]
    cs           = max(20, int(len(X_tr) * 0.20))
    X_fit, y_fit = X_tr[:-cs], y_tr[:-cs]
    X_cal, y_cal = X_tr[-cs:], y_tr[-cs:]
    pos_r = y_fit.mean()
    spw   = (1 - pos_r) / pos_r if pos_r > 0 else 1.0
    base  = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="auc",
        random_state=42, verbosity=0,
    )
    base.fit(X_fit, y_fit)
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    cal.fit(X_cal, y_cal)
    fold_models[ylbl] = cal
    print(f"  Fold {ylbl}: n={mask.sum():,}  (train {tr_s[:4]}~{tr_e[:4]})")


# ═══════════════════════════════════════════════════════════════════
# 4. 신호 사전 계산
# ═══════════════════════════════════════════════════════════════════
print("\n신호 계산...")
all_days = _strip_tz(kospi_raw.index)
bt_days  = all_days[(all_days >= BT_START) & (all_days <= BT_END)]

def _get_model(date):
    return fold_models.get(str(pd.Timestamp(date).year))

# 통합 신호: {date: [{ticker, agent, win_prob(opt), atr}, ...]}
signals_by_date: dict[pd.Timestamp, list[dict]] = {}

# ── 리버전 신호 ──────────────────────────────────────────────────
n_rev = 0
for t, df_f in feat_data.items():
    sub    = df_f.dropna(subset=FC_REV)
    sub_bt = sub[(sub.index >= BT_START) & (sub.index < BT_END)]
    if sub_bt.empty: continue
    det = detect_reversion_rows(sub_bt).reindex(sub_bt.index, fill_value=False)
    for date in sub_bt[det].index:
        model = _get_model(date)
        if model is None: continue
        Xp  = sub_bt.loc[date, FC_REV].values.reshape(1, -1).astype("float32")
        wp  = float(model.predict_proba(Xp)[0, 1])
        if wp < REV_WP: continue
        atr_v = 0.0
        di    = ind_data.get(t)
        if di is not None:
            ir = _row(di, date)
            if not ir.empty:
                atr_v = float(ir["atr"].iloc[0]) if not pd.isna(ir["atr"].iloc[0]) else 0.0
        ts = _to_ts(date)
        signals_by_date.setdefault(ts, []).append(
            {"ticker": t, "agent": "reversion", "win_prob": wp, "atr": atr_v}
        )
        n_rev += 1
print(f"  리버전: {n_rev}건")

# ── 트렌드 신호 ──────────────────────────────────────────────────
n_tr = 0
for t, df_i in ind_data.items():
    idx = _strip_tz(df_i.index)
    for date in bt_days:
        p = np.where(idx == date)[0]
        if len(p) == 0: continue
        r = df_i.iloc[p[0]]
        need = ["adx","ma5","ma20","ma60","ma200","vol_ma20"]
        if any(pd.isna(r.get(c, np.nan)) for c in need): continue
        if r["adx"] < TR_ADX: continue
        if not (r["ma5"] > r["ma20"] > r["ma60"] > r["ma200"]): continue
        if r["vol_ma20"] == 0 or r["Volume"] < r["vol_ma20"] * TR_VOL: continue
        km200 = kospi_ma200.get(date, np.nan)
        kc    = kospi_close.get(date, np.nan)
        if pd.isna(km200) or pd.isna(kc) or kc <= km200: continue
        ts = _to_ts(date)
        # 같은 종목에 리버전 신호가 이미 있으면 skip (reversion 우선)
        existing = signals_by_date.get(ts, [])
        if any(s["ticker"] == t and s["agent"] == "reversion" for s in existing):
            continue
        signals_by_date.setdefault(ts, []).append(
            {"ticker": t, "agent": "trend", "atr": float(r["atr"])}
        )
        n_tr += 1
print(f"  트렌드: {n_tr}건")


# ═══════════════════════════════════════════════════════════════════
# 5. 시뮬레이션 함수
# ═══════════════════════════════════════════════════════════════════
def run_sim(bt_days, sigs, mode="both", rev_slots=10, tr_slots=10, shared=True):
    """
    shared=True : rev_slots+tr_slots 합산 상한, 에이전트 간 슬롯 공유
    shared=False: reversion 전용 rev_slots / trend 전용 tr_slots, 서로 침범 불가
    """
    capital   = 1.0
    positions = []
    trades    = []
    daily_nav = []        # (date, nav)
    daily_cash = []       # (date, cash_ratio)

    for i, today in enumerate(bt_days):
        next_day = bt_days[i+1] if i+1 < len(bt_days) else None

        # 익일 시초가 청산
        still = []
        for pos in positions:
            if pos.get("_exit_next"):
                r   = _row(raw_data.get(pos["ticker"]), today)
                if r.empty: pos.pop("_exit_next", None); still.append(pos); continue
                ep  = pos["entry_price"]
                xp  = float(r["Open"].iloc[0])
                net = xp / ep - 1 - COST_RT
                capital += pos["size"] * (1 + net)
                trades.append({
                    "net_ret": net, "exit_reason": pos["_exit_reason"],
                    "win": net > 0, "hold_days": pos["hold_days"], "agent": pos["agent"],
                })
            else:
                still.append(pos)
        positions = still

        # 청산 조건 체크
        still = []
        for pos in positions:
            pos["hold_days"] += 1
            r  = _row(raw_data.get(pos["ticker"]), today)
            ri = _row(ind_data.get(pos["ticker"]),  today)
            if r.empty: still.append(pos); continue
            hi, lo, cl = (float(r[c].iloc[0]) for c in ("High","Low","Close"))
            ep  = pos["entry_price"]
            pos["high_since_entry"] = max(pos.get("high_since_entry", ep), hi)

            rsn = None
            if pos["agent"] == "reversion":
                if   hi >= ep * (1 + REV_TP):     rsn = "TP"
                elif lo <= ep * (1 - REV_SL):     rsn = "SL"
                elif pos["hold_days"] >= REV_HOLD: rsn = "TIME"
            else:
                atr_e    = pos.get("atr_at_entry", 0)
                trail_px = pos["high_since_entry"] - TR_TRAIL * atr_e if atr_e > 0 else -np.inf
                if cl <= trail_px:
                    rsn = "TRAIL"
                elif not ri.empty:
                    ma20 = ri["ma20"].iloc[0] if "ma20" in ri.columns else np.nan
                    adxv = ri["adx"].iloc[0]  if "adx"  in ri.columns else 99.0
                    if not pd.isna(ma20) and cl < ma20:          rsn = "MA20"
                    elif not pd.isna(adxv) and adxv < ADX_EXIT:  rsn = "ADX"
                if rsn is None and pos["hold_days"] >= TR_HOLD:  rsn = "TIME"

            if rsn and next_day is not None:
                pos["_exit_next"] = True; pos["_exit_reason"] = rsn
            still.append(pos)
        positions = still

        # 신규 진입
        if next_day is not None and today in sigs:
            day_sigs = sigs[today]
            if mode == "reversion": day_sigs = [s for s in day_sigs if s["agent"] == "reversion"]
            elif mode == "trend":   day_sigs = [s for s in day_sigs if s["agent"] == "trend"]
            day_sigs = sorted(day_sigs, key=lambda s: -s.get("win_prob", 0))

            rev_cnt = sum(1 for p in positions if p["agent"] == "reversion")
            tr_cnt  = sum(1 for p in positions if p["agent"] == "trend")

            for sig in day_sigs:
                ag = sig["agent"]
                # 슬롯 체크
                if shared:
                    if len(positions) >= (rev_slots + tr_slots): break
                else:
                    if ag == "reversion" and rev_cnt >= rev_slots: continue
                    if ag == "trend"     and tr_cnt  >= tr_slots:  continue

                t = sig["ticker"]
                if any(p["ticker"] == t for p in positions): continue
                r2 = _row(raw_data.get(t), next_day)
                if r2.empty: continue
                ep    = float(r2["Open"].iloc[0])
                if ep <= 0: continue
                atr_v = sig["atr"]
                sz    = (calc_rev_size(capital, sig["win_prob"], atr_v, ep)
                         if ag == "reversion"
                         else calc_tr_size(capital, atr_v, ep))
                if sz <= 0: continue
                capital -= sz * (1 + COST_RT / 2)
                positions.append({
                    "ticker": t, "agent": ag,
                    "entry_price": ep, "hold_days": 0, "size": sz,
                    "atr_at_entry": atr_v, "high_since_entry": ep,
                })
                if ag == "reversion": rev_cnt += 1
                else:                  tr_cnt  += 1

        # 일별 NAV + 현금 비율
        pm = 0.0
        for pos in positions:
            r2 = _row(raw_data.get(pos["ticker"]), today)
            cp = float(r2["Close"].iloc[0]) if not r2.empty else pos["entry_price"]
            pm += pos["size"] * (cp / pos["entry_price"])
        nav_today = capital + pm
        daily_nav.append((today, nav_today))
        daily_cash.append((today, capital / nav_today if nav_today > 0 else 1.0))

    if not trades or len(daily_nav) < 2:
        return {"n_trades": 0}

    df_tr  = pd.DataFrame(trades)
    nav    = pd.Series([v for _, v in daily_nav],
                       index=pd.DatetimeIndex([d for d, _ in daily_nav]))
    cash_s = pd.Series([v for _, v in daily_cash],
                       index=pd.DatetimeIndex([d for d, _ in daily_cash]))
    dr     = nav.pct_change().dropna()
    wins   = df_tr[df_tr["net_ret"] > 0]["net_ret"]
    losses = df_tr[df_tr["net_ret"] <= 0]["net_ret"]

    annual = {}
    cash_annual = {}
    for y in ["2024","2025","2026"]:
        sub = nav[nav.index.year == int(y)]
        if len(sub) >= 2:
            annual[y] = round((sub.iloc[-1] / sub.iloc[0] - 1) * 100, 2)
        sc = cash_s[cash_s.index.year == int(y)]
        if len(sc) > 0:
            cash_annual[y] = round(float(sc.mean()) * 100, 1)

    monthly = {}
    prev = nav.iloc[0]
    for period, grp in nav.resample("ME"):
        if grp.empty: continue
        cur = grp.iloc[-1]
        monthly[period.strftime("%Y-%m")] = round((cur / prev - 1) * 100, 2)
        prev = cur

    return {
        "n_trades":   len(df_tr),
        "total_ret":  round((nav.iloc[-1] - 1) * 100, 2),
        "sharpe":     round(dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0, 3),
        "mdd":       round(((nav - nav.cummax()) / nav.cummax()).min() * 100, 2),
        "win_rate":  round(df_tr["win"].mean() * 100, 1),
        "avg_hold":  round(df_tr["hold_days"].mean(), 1),
        "pf":        round(abs(wins.mean() / losses.mean()), 2)
                     if len(losses) > 0 and losses.mean() != 0 else 0.0,
        "exit":       df_tr["exit_reason"].value_counts().to_dict(),
        "agent_cnt":  df_tr["agent"].value_counts().to_dict(),
        "annual":     annual,
        "monthly":    monthly,
        "cash_annual": cash_annual,
    }


# ═══════════════════════════════════════════════════════════════════
# 6. 실행 (5가지: 단독 2개 + 합산 3가지 슬롯 설정)
# ═══════════════════════════════════════════════════════════════════
print(f"\n백테스트 실행 ({BT_START} ~ {BT_END})...")

print("  reversion 단독...")
r_rev = run_sim(bt_days, signals_by_date, mode="reversion",
                rev_slots=10, tr_slots=0, shared=False)
print(f"    → {r_rev.get('total_ret', 0):+.2f}%  거래={r_rev.get('n_trades', 0)}건")

print("  trend 단독...")
r_tr = run_sim(bt_days, signals_by_date, mode="trend",
               rev_slots=0, tr_slots=10, shared=False)
print(f"    → {r_tr.get('total_ret', 0):+.2f}%  거래={r_tr.get('n_trades', 0)}건")

print("  합산 공유 10슬롯 (기존)...")
r_s10 = run_sim(bt_days, signals_by_date, mode="both",
                rev_slots=10, tr_slots=10, shared=True)
print(f"    → {r_s10.get('total_ret', 0):+.2f}%  거래={r_s10.get('n_trades', 0)}건")

print("  합산 분리 5+5...")
r_55 = run_sim(bt_days, signals_by_date, mode="both",
               rev_slots=5, tr_slots=5, shared=False)
print(f"    → {r_55.get('total_ret', 0):+.2f}%  거래={r_55.get('n_trades', 0)}건")

print("  합산 분리 10+10...")
r_1010 = run_sim(bt_days, signals_by_date, mode="both",
                 rev_slots=10, tr_slots=10, shared=False)
print(f"    → {r_1010.get('total_ret', 0):+.2f}%  거래={r_1010.get('n_trades', 0)}건")


# ═══════════════════════════════════════════════════════════════════
# 7. 결과 출력
# ═══════════════════════════════════════════════════════════════════
YEARS = ["2024","2025","2026"]

configs = [
    ("공유 10슬롯(기존)", r_s10),
    ("분리 5+5",          r_55),
    ("분리 10+10(이번)",  r_1010),
]
W = 16  # 열 너비

print(f"\n{'='*72}")
print(f"합산 백테스트 슬롯 비교  ({BT_START} ~ {BT_END})")
print(f"{'='*72}")

# ── 연도별 수익률 ────────────────────────────────────────────────
print(f"\n[연도별 수익률]")
hdr = f"  {'':>6}" + "".join(f"  {lbl:>{W}}" for lbl, _ in configs)
print(hdr)
print("  " + "-"*(8 + (W+2)*3))
for y in YEARS:
    row = f"  {y:>6}"
    for _, r in configs:
        v = r.get("annual",{}).get(y, 0)
        row += f"  {v:>+{W-1}.2f}%"
    print(row)

# ── 전체 지표 ────────────────────────────────────────────────────
print(f"\n[전체 지표]")
metrics = [
    ("수익률",  "total_ret", lambda v: f"{v:+.2f}%"),
    ("샤프",    "sharpe",    lambda v: f"{v:.3f}"),
    ("MDD",     "mdd",       lambda v: f"{v:.2f}%"),
    ("거래(합산)","n_trades", lambda v: f"{v}건"),
    ("승률",    "win_rate",  lambda v: f"{v:.1f}%"),
    ("손익비",  "pf",        lambda v: f"{v:.2f}"),
]
hdr = f"  {'':>12}" + "".join(f"  {lbl:>{W}}" for lbl, _ in configs)
print(hdr)
print("  " + "-"*(14 + (W+2)*3))
for label, key, fmt_fn in metrics:
    row = f"  {label:>12}"
    for _, r in configs:
        v = r.get(key, 0)
        row += f"  {fmt_fn(v):>{W}}"
    print(row)

# ── 에이전트별 거래 수 ────────────────────────────────────────────
print(f"\n[에이전트별 거래 수 / 비율]")
hdr = f"  {'':>16}" + "".join(f"  {lbl:>{W}}" for lbl, _ in configs)
print(hdr)
print("  " + "-"*(18 + (W+2)*3))
for ag in ["reversion","trend"]:
    row = f"  {ag:>16}"
    for _, r in configs:
        cnt   = r.get("agent_cnt",{}).get(ag, 0)
        total = sum(r.get("agent_cnt",{}).values()) or 1
        row  += f"  {cnt}건 ({cnt/total*100:.0f}%):>{W-12}"
        row   = row  # 아래에서 포매팅
    # 재포매팅
    row = f"  {ag:>16}"
    for _, r in configs:
        cnt   = r.get("agent_cnt",{}).get(ag, 0)
        total = sum(r.get("agent_cnt",{}).values()) or 1
        cell  = f"{cnt}건({cnt/total*100:.0f}%)"
        row  += f"  {cell:>{W}}"
    print(row)

# ── 월별 상관계수 ─────────────────────────────────────────────────
print(f"\n[월별 수익률 상관계수  (reversion vs trend)]")
hdr = f"  {'':>10}" + "".join(f"  {lbl:>{W}}" for lbl, _ in configs)
print(hdr)
print("  " + "-"*(12 + (W+2)*3))
row = f"  {'corr':>10}"
for _, r in configs:
    rv_mon = r_rev.get("monthly", {})
    tr_mon = r_tr.get("monthly",  {})
    # 합산 결과의 월 데이터로 rev/tr 분리 계산은 어렵지만
    # reversion 단독 / trend 단독 월수익률 공통 구간으로 계산
    common = sorted(set(rv_mon) & set(tr_mon))
    if len(common) >= 3:
        rv_v = [rv_mon[m] for m in common]
        tr_v = [tr_mon[m] for m in common]
        c = float(np.corrcoef(rv_v, tr_v)[0, 1])
        row += f"  {c:>{W}.3f}"
    else:
        row += f"  {'N/A':>{W}}"
print(row)
# 상관계수는 슬롯 설정에 무관 (단독 전략의 월수익률로 계산)
common = sorted(set(r_rev.get("monthly",{}).keys()) & set(r_tr.get("monthly",{}).keys()))
if len(common) >= 3:
    rv_v = [r_rev["monthly"][m] for m in common]
    tr_v = [r_tr["monthly"][m]  for m in common]
    corr = float(np.corrcoef(rv_v, tr_v)[0, 1])
    interp = ("역상관" if corr < 0 else
              "약한 양의 상관" if corr < 0.3 else
              "중간 상관" if corr < 0.6 else "강한 양의 상관")
    print(f"  → {corr:.3f}  ({interp}, {len(common)}개월 기준)")

# ── 연도별 평균 현금 보유 비율 ────────────────────────────────────
print(f"\n[연도별 평균 현금 보유 비율 (%)]")
hdr = f"  {'':>6}" + "".join(f"  {lbl:>{W}}" for lbl, _ in configs)
print(hdr)
print("  " + "-"*(8 + (W+2)*3))
for y in YEARS:
    row = f"  {y:>6}"
    for _, r in configs:
        v = r.get("cash_annual",{}).get(y, 0)
        row += f"  {v:>{W}.1f}%"
    print(row)

# ── 단독 전략 참고 ────────────────────────────────────────────────
print(f"\n[단독 전략 참고]")
fmt2 = "  {:<20s}  {:>8}  {:>7}  {:>8}  {:>6}  {:>7}"
print(fmt2.format("전략","수익률","샤프","MDD","거래","승률"))
print("  " + "-"*60)
for lbl, r in [("reversion 단독", r_rev), ("trend 단독", r_tr)]:
    if r.get("n_trades", 0) == 0: continue
    print(fmt2.format(
        lbl,
        f"{r['total_ret']:+.2f}%", f"{r['sharpe']:.3f}",
        f"{r['mdd']:.2f}%", f"{r['n_trades']}건",
        f"{r['win_rate']:.1f}%",
    ))
