#!/usr/bin/env python3
"""
combined_backtest_v2.py — Rolling 3년 × PIT 유니버스 3-way 비교 백테스트

A) Expanding + 정적 200  (기존 방식, 2020년 데이터부터)
B) Rolling 3년 + 정적 200  (학습방식만 변경 효과 분리)
C) Rolling 3년 + PIT 200  (FDR 근사 PIT — 생존편향 부분 제거)

검증 구간: 2023 / 2024 / 2025 / 2026

PIT 유니버스 구성:
  - FDR 현재 시총 상위 500개 종목 풀 확보
  - 각 연도 기준일(전년말)에서 yfinance 종가 × FDR 발행주식수 → 시총 추정
  - 연도별 상위 200개 선정 (현재 목록 기준 근사치)
  - 한계: 상폐 종목 미포함, 발행주식수는 현재 기준 고정

Rolling 3년:
  2023 검증: train 2020~2022
  2024 검증: train 2021~2023
  2025 검증: train 2022~2024
  2026 검증: train 2023~2025

전략 파라미터 (변경 없음):
  - reversion: win_prob>=0.52 / TP=15% / SL=8% / 10일 / 10슬롯
  - trend: ADX>=25 / MA정배열 / vol>1.3x / trailing 2.0ATR / MA20 이탈 / 10슬롯
  - 슬롯 분리 10+10, 단일 종목 20% 캡, 하프 켈리 + ATR 사이징
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo-root: 하위 폴더에서 직접 실행 대비

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

from ml.features import (
    add_features, _triple_barrier_pnl,
    detect_reversion_rows, FEATURE_COLS_REVERSION,
)
from config import ML_MIN_WIN_PROB, ML_MIN_RISK_REWARD

# ═══════════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════════
BT_START = "2023-01-01"
BT_END   = "2026-06-20"

REV_TP, REV_SL, REV_HOLD = 0.15, 0.08, 10
REV_WP  = ML_MIN_WIN_PROB      # config 단일 진실 소스 — 라이브 _eval_agent와 동일 임계값
REV_RR  = ML_MIN_RISK_REWARD   # 손익비 게이트 — 라이브 _eval_agent와 동일
REV_PF  = REV_TP / REV_SL

# 손익비 계산용 모델 메트릭 (라이브 _eval_agent와 동일한 avg_win/avg_loss 사용)
def _load_rev_metrics():
    try:
        from ml.model import load_model
        _, m = load_model("_global", "reversion")
        return m.get("avg_win", 0.099), m.get("avg_loss", 0.065)
    except Exception:
        return 0.099, 0.065
_REV_AVG_WIN, _REV_AVG_LOSS = _load_rev_metrics()

TR_ADX, TR_TRAIL, TR_VOL = 25, 2.0, 1.3
TR_HOLD  = 60
ADX_EXIT = 20

MAX_POS   = 10
MAX_ALLOC = 0.20
ATR_RISK  = 0.01
COST_RT   = 0.0046

DATA_START = "2019-01-01"   # Rolling 3년 학습을 위해 2020 앞 버퍼 포함

FC_REV = FEATURE_COLS_REVERSION

# ── WF Fold 정의 ──────────────────────────────────────────────────
#  (train_start, train_end, val_year)
WF_FOLDS_EXPANDING = [
    ("2020-01-01", "2023-01-01", "2023"),
    ("2020-01-01", "2024-01-01", "2024"),
    ("2020-01-01", "2025-01-01", "2025"),
    ("2020-01-01", "2026-01-01", "2026"),
]

WF_FOLDS_ROLLING = [
    ("2020-01-01", "2023-01-01", "2023"),
    ("2021-01-01", "2024-01-01", "2024"),
    ("2022-01-01", "2025-01-01", "2025"),
    ("2023-01-01", "2026-01-01", "2026"),
]

# PIT 유니버스 기준일 (전년 마지막 거래일 근사)
PIT_REF_DATES = {
    "2023": "2022-12-29",
    "2024": "2023-12-28",
    "2025": "2024-12-27",
    "2026": "2025-12-26",
}

STATIC_TOP_N = 200
PIT_POOL_N   = 500   # FDR 시총 상위 N개를 PIT 후보 풀로 사용


# ═══════════════════════════════════════════════════════════════════
# 유틸
# ═══════════════════════════════════════════════════════════════════
def _strip_tz(idx):
    return idx.tz_localize(None) if idx.tzinfo else idx

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
    df["tr"]       = tr
    df["atr"]      = tr.ewm(com=13, adjust=False).mean()
    df["vol_ma20"] = v.rolling(20).mean()
    dm_p = (h - h.shift(1)).clip(lower=0)
    dm_m = (l.shift(1) - l).clip(lower=0)
    dm_p = dm_p.where(dm_p >= dm_m, 0.0)
    dm_m = dm_m.where(dm_m >  dm_p, 0.0)
    atr_w = tr.ewm(com=13, adjust=False).mean()
    di_p  = 100 * dm_p.ewm(com=13, adjust=False).mean() / atr_w
    di_m  = 100 * dm_m.ewm(com=13, adjust=False).mean() / atr_w
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
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
# PIT 유니버스 구성
# ═══════════════════════════════════════════════════════════════════
def build_pit_universe(raw_data: dict, fdr_df: pd.DataFrame,
                       ref_date: str, top_n: int = 200) -> list[str]:
    """
    FDR 발행주식수(현재) × yfinance 과거 종가 → 시총 추정 → 상위 top_n 반환
    한계: 상폐 종목 미포함, 발행주식수는 현재 기준 고정
    """
    ref_ts = _to_ts(ref_date)
    caps = {}
    for _, row in fdr_df.iterrows():
        ticker = row["yf_code"]
        shares = row.get("Stocks", 0)
        if shares <= 0:
            continue
        df = raw_data.get(ticker)
        if df is None or df.empty:
            continue
        valid = df[df.index <= ref_ts]
        if valid.empty:
            continue
        close = float(valid["Close"].iloc[-1])
        caps[ticker] = close * shares

    sorted_caps = sorted(caps.items(), key=lambda x: -x[1])
    return [t for t, _ in sorted_caps[:top_n]]


# ═══════════════════════════════════════════════════════════════════
# 0. FDR 종목 목록 확보
# ═══════════════════════════════════════════════════════════════════
print("FDR 시총 상위 500개 목록 로드...")
import FinanceDataReader as fdr
fdr_raw = fdr.StockListing("KRX")
fdr_raw = fdr_raw.sort_values("Marcap", ascending=False).head(PIT_POOL_N).copy()
fdr_raw["yf_code"] = fdr_raw.apply(
    lambda r: r["Code"] + ".KS" if r["MarketId"] == "STK" else r["Code"] + ".KQ", axis=1
)
fdr_raw = fdr_raw.reset_index(drop=True)
pool_tickers = fdr_raw["yf_code"].tolist()
print(f"  풀 종목: {len(pool_tickers)}개")

# 정적 200개 (현재 기준)
static_200 = pool_tickers[:STATIC_TOP_N]


# ═══════════════════════════════════════════════════════════════════
# 1. KOSPI + 전종목 데이터 다운로드
# ═══════════════════════════════════════════════════════════════════
print(f"\nKOSPI 다운로드 ({DATA_START}~)...")
kospi_raw = yf.download("^KS11", start=DATA_START, auto_adjust=True, progress=False)
if isinstance(kospi_raw.columns, pd.MultiIndex):
    kospi_raw.columns = kospi_raw.columns.get_level_values(0)
kospi_raw.index = _strip_tz(kospi_raw.index)
kospi_ind     = compute_indicators(kospi_raw)
kospi_ma200   = kospi_ind["ma200"].dropna()
kospi_close   = kospi_raw["Close"]
print(f"  KOSPI {len(kospi_raw)}행")

print(f"\n{len(pool_tickers)}개 종목 다운로드 (약 5~10분 소요)...")
raw_data: dict[str, pd.DataFrame] = {}
BATCH = 50
for i in range(0, len(pool_tickers), BATCH):
    batch = pool_tickers[i:i+BATCH]
    try:
        df_b = yf.download(
            batch, start=DATA_START,
            auto_adjust=True, progress=False, threads=True,
        )
        if isinstance(df_b.columns, pd.MultiIndex):
            for t in batch:
                try:
                    sub = df_b.xs(t, axis=1, level=1)[["Open","High","Low","Close","Volume"]]
                    sub = sub.dropna(how="all")
                    sub.index = _strip_tz(sub.index)
                    if len(sub) >= 100:
                        raw_data[t] = sub
                except Exception:
                    pass
        else:
            for t in batch:
                raw_data[t] = df_b[["Open","High","Low","Close","Volume"]].dropna()
    except Exception as e:
        print(f"  배치 {i//BATCH+1} 실패: {e}")
    if (i // BATCH + 1) % 2 == 0:
        print(f"  진행: {min(i+BATCH, len(pool_tickers))}/{len(pool_tickers)}개 완료")

print(f"  다운로드 완료: {len(raw_data)}개 / {len(pool_tickers)}개")


# ═══════════════════════════════════════════════════════════════════
# 2. 피처 + 지표 계산
# ═══════════════════════════════════════════════════════════════════
print("\n피처 + 지표 계산...")
feat_data: dict[str, pd.DataFrame] = {}
ind_data:  dict[str, pd.DataFrame] = {}

for t, df in raw_data.items():
    try:
        df_f = add_features(df, kospi_df=kospi_raw)
        df_f.index = _strip_tz(df_f.index)
        if len(df_f.dropna(subset=FC_REV)) >= 50:
            feat_data[t] = df_f
    except Exception:
        pass
    try:
        df_i = compute_indicators(df)
        if len(df_i.dropna(subset=["ma200", "adx"])) >= 20:
            ind_data[t] = df_i
    except Exception:
        pass

print(f"  리버전 피처: {len(feat_data)}개  |  트렌드 지표: {len(ind_data)}개")


# ═══════════════════════════════════════════════════════════════════
# 3. PIT 유니버스 사전 구성
# ═══════════════════════════════════════════════════════════════════
print("\nPIT 유니버스 구성...")
pit_universes: dict[str, list[str]] = {}
pit_universe_sets: dict[str, set] = {}
for year, ref_date in PIT_REF_DATES.items():
    pit_universes[year] = build_pit_universe(raw_data, fdr_raw, ref_date, top_n=STATIC_TOP_N)
    pit_universe_sets[year] = set(pit_universes[year])
    print(f"  {year} PIT 유니버스 (기준일 {ref_date}): {len(pit_universes[year])}개")

# 누적 PIT 종목 수
all_pit = set()
for v in pit_universes.values():
    all_pit.update(v)
print(f"  PIT 누적 등장 종목: {len(all_pit)}개")

# 정적 vs PIT 차이
static_set = set(static_200)
print(f"\n  정적 200 vs PIT 평균 겹침 비율:")
for year, pit_set in pit_universe_sets.items():
    overlap = len(static_set & pit_set)
    print(f"    {year}: {overlap}/200개 공통 ({overlap/2:.0f}%)")


# ═══════════════════════════════════════════════════════════════════
# 4. 라벨 + 피처 데이터셋 구축 함수
# ═══════════════════════════════════════════════════════════════════
def build_rev_dataset(ticker_set: set | None = None) -> tuple:
    """리버전 학습용 (X, y, dates) 구축. ticker_set=None → 전체"""
    rev_rows = []
    for t, df_f in feat_data.items():
        if ticker_set is not None and t not in ticker_set:
            continue
        try:
            labels_arr, _ = _triple_barrier_pnl(
                df_f, tp_pct=REV_TP, sl_pct=REV_SL, max_holding_days=REV_HOLD
            )
            labels_s = pd.Series(labels_arr, index=df_f.index)
            det_mask = detect_reversion_rows(df_f).reindex(df_f.index, fill_value=False)
            sub      = df_f.dropna(subset=FC_REV)
            det_sub  = det_mask.reindex(sub.index, fill_value=False)
            lbl_sub  = labels_s.reindex(sub.index)
            valid    = det_sub & lbl_sub.notna()
            sub2     = sub[valid].copy()
            if len(sub2) < 5:
                continue
            sub2["_label"]  = lbl_sub[valid].values
            sub2["_ticker"] = t
            sub2["_date"]   = sub2.index
            rev_rows.append(sub2)
        except Exception:
            pass
    if not rev_rows:
        return np.empty((0, len(FC_REV))), np.empty(0), pd.DatetimeIndex([])
    rev_df = pd.concat(rev_rows).sort_values("_date").reset_index(drop=True)
    X = rev_df[FC_REV].values.astype("float32")
    y = rev_df["_label"].values.astype(int)
    d = pd.to_datetime(rev_df["_date"].values)
    return X, y, d


# ═══════════════════════════════════════════════════════════════════
# 5. 모델 학습 함수
# ═══════════════════════════════════════════════════════════════════
def train_fold_models(wf_folds, X_all, y_all, d_all) -> dict:
    """WF fold별 XGBoost 학습 (Raw, Platt Scaling 없음)"""
    fold_models = {}
    for (tr_s, tr_e, ylbl) in wf_folds:
        mask = (d_all >= tr_s) & (d_all < tr_e)
        if mask.sum() < 50:
            print(f"    Fold {ylbl}: 샘플 부족 ({mask.sum()})")
            fold_models[ylbl] = None
            continue
        X_tr, y_tr = X_all[mask], y_all[mask]
        pos_r = y_tr.mean()
        spw   = (1 - pos_r) / pos_r if pos_r > 0 else 1.0
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="auc",
            random_state=42, verbosity=0,
        )
        model.fit(X_tr, y_tr)
        fold_models[ylbl] = model
        print(f"    Fold {ylbl}: n={mask.sum():,}  train {tr_s[:7]}~{tr_e[:7]}")
    return fold_models


# ═══════════════════════════════════════════════════════════════════
# 6. 신호 사전 계산 함수
# ═══════════════════════════════════════════════════════════════════
def build_signals(fold_models, universe_by_year: dict | None = None):
    """
    universe_by_year = None → 정적 200 사용
                    = {"2023": [tickers], ...} → PIT 사용
    """
    all_days = _strip_tz(kospi_raw.index)
    bt_days  = all_days[(all_days >= BT_START) & (all_days <= BT_END)]
    sigs: dict[pd.Timestamp, list[dict]] = {}

    def _get_model(date):
        return fold_models.get(str(pd.Timestamp(date).year))

    def _in_universe(ticker, date):
        if universe_by_year is None:
            return ticker in static_set
        yr = str(pd.Timestamp(date).year)
        return ticker in pit_universe_sets.get(yr, set())

    # 리버전 신호
    n_rev = 0
    for t, df_f in feat_data.items():
        sub    = df_f.dropna(subset=FC_REV)
        sub_bt = sub[(sub.index >= BT_START) & (sub.index < BT_END)]
        if sub_bt.empty:
            continue
        det = detect_reversion_rows(sub_bt).reindex(sub_bt.index, fill_value=False)
        for date in sub_bt[det].index:
            if not _in_universe(t, date):
                continue
            model = _get_model(date)
            if model is None:
                continue
            Xp = sub_bt.loc[date, FC_REV].values.reshape(1, -1).astype("float32")
            wp = float(model.predict_proba(Xp)[0, 1])
            if wp < REV_WP:
                continue
            # 손익비 게이트 — 라이브 _eval_agent와 동일
            _ev_win  = wp * _REV_AVG_WIN
            _ev_loss = (1 - wp) * _REV_AVG_LOSS
            if _ev_loss > 0 and (_ev_win / _ev_loss) < REV_RR:
                continue
            atr_v = 0.0
            di = ind_data.get(t)
            if di is not None:
                ir = _row(di, date)
                if not ir.empty:
                    atr_v = float(ir["atr"].iloc[0]) if not pd.isna(ir["atr"].iloc[0]) else 0.0
            ts = _to_ts(date)
            sigs.setdefault(ts, []).append(
                {"ticker": t, "agent": "reversion", "win_prob": wp, "atr": atr_v}
            )
            n_rev += 1

    # 트렌드 신호
    n_tr = 0
    for t, df_i in ind_data.items():
        idx = _strip_tz(df_i.index)
        for date in bt_days:
            if not _in_universe(t, date):
                continue
            p = np.where(idx == date)[0]
            if len(p) == 0:
                continue
            r = df_i.iloc[p[0]]
            need = ["adx", "ma5", "ma20", "ma60", "ma200", "vol_ma20"]
            if any(pd.isna(r.get(c, np.nan)) for c in need):
                continue
            if r["adx"] < TR_ADX:
                continue
            if not (r["ma5"] > r["ma20"] > r["ma60"] > r["ma200"]):
                continue
            if r["Close"] <= r["ma20"]:   # 종가가 ma20 위일 때만 진입(청산 규칙 cl<ma20과 짝)
                continue
            if r["vol_ma20"] == 0 or r["Volume"] < r["vol_ma20"] * TR_VOL:
                continue
            km200 = kospi_ma200.get(date, np.nan)
            kc    = kospi_close.get(date, np.nan)
            if pd.isna(km200) or pd.isna(kc) or kc <= km200:
                continue
            ts = _to_ts(date)
            existing = sigs.get(ts, [])
            if any(s["ticker"] == t and s["agent"] == "reversion" for s in existing):
                continue
            sigs.setdefault(ts, []).append(
                {"ticker": t, "agent": "trend", "atr": float(r["atr"])}
            )
            n_tr += 1

    return sigs, bt_days, n_rev, n_tr


# ═══════════════════════════════════════════════════════════════════
# 7. 시뮬레이션 함수 (슬롯 분리 10+10)
# ═══════════════════════════════════════════════════════════════════
def run_sim(bt_days, sigs, rev_slots=10, tr_slots=10):
    capital   = 1.0
    positions = []
    trades    = []
    daily_nav = []

    for i, today in enumerate(bt_days):
        next_day = bt_days[i+1] if i+1 < len(bt_days) else None

        # 익일 시초가 청산
        still = []
        for pos in positions:
            if pos.get("_exit_next"):
                r = _row(raw_data.get(pos["ticker"]), today)
                if r.empty:
                    pos.pop("_exit_next", None)
                    still.append(pos)
                    continue
                ep  = pos["entry_price"]
                xp  = float(r["Open"].iloc[0])
                net = xp / ep - 1 - COST_RT
                capital += pos["size"] * (1 + net)
                trades.append({
                    "net_ret": net,
                    "exit_reason": pos["_exit_reason"],
                    "win": net > 0,
                    "hold_days": pos["hold_days"],
                    "agent": pos["agent"],
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
            if r.empty:
                still.append(pos)
                continue
            hi, lo, cl = (float(r[c].iloc[0]) for c in ("High", "Low", "Close"))
            ep = pos["entry_price"]
            pos["high_since_entry"] = max(pos.get("high_since_entry", ep), hi)

            rsn = None
            if pos["agent"] == "reversion":
                if   lo <= ep * (1 - REV_SL):     rsn = "SL"   # SL 우선 — _triple_barrier_pnl 라벨·walkforward·paper와 동일
                elif hi >= ep * (1 + REV_TP):     rsn = "TP"
                elif pos["hold_days"] >= REV_HOLD: rsn = "TIME"
            else:
                atr_e    = pos.get("atr_at_entry", 0)
                trail_px = pos["high_since_entry"] - TR_TRAIL * atr_e if atr_e > 0 else -np.inf
                if cl <= trail_px:
                    rsn = "TRAIL"
                elif not ri.empty:
                    ma20 = ri["ma20"].iloc[0] if "ma20" in ri.columns else np.nan
                    adxv = ri["adx"].iloc[0]  if "adx"  in ri.columns else 99.0
                    if not pd.isna(ma20) and cl < ma20:         rsn = "MA20"
                    elif not pd.isna(adxv) and adxv < ADX_EXIT: rsn = "ADX"
                if rsn is None and pos["hold_days"] >= TR_HOLD: rsn = "TIME"

            if rsn and next_day is not None:
                pos["_exit_next"]   = True
                pos["_exit_reason"] = rsn
            still.append(pos)
        positions = still

        # 신규 진입
        if next_day is not None and today in sigs:
            day_sigs = sorted(sigs[today], key=lambda s: -s.get("win_prob", 0))
            rev_cnt  = sum(1 for p in positions if p["agent"] == "reversion")
            tr_cnt   = sum(1 for p in positions if p["agent"] == "trend")

            for sig in day_sigs:
                ag = sig["agent"]
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

        # 일별 NAV
        pm = 0.0
        for pos in positions:
            r2 = _row(raw_data.get(pos["ticker"]), today)
            cp = float(r2["Close"].iloc[0]) if not r2.empty else pos["entry_price"]
            pm += pos["size"] * (cp / pos["entry_price"])
        daily_nav.append((today, capital + pm))

    if not trades or len(daily_nav) < 2:
        return {"n_trades": 0}

    df_tr = pd.DataFrame(trades)
    nav   = pd.Series(
        [v for _, v in daily_nav],
        index=pd.DatetimeIndex([d for d, _ in daily_nav])
    )
    dr   = nav.pct_change().dropna()
    wins = df_tr[df_tr["net_ret"] > 0]["net_ret"]
    loss = df_tr[df_tr["net_ret"] <= 0]["net_ret"]

    annual = {}
    for y in ["2023", "2024", "2025", "2026"]:
        sub = nav[nav.index.year == int(y)]
        if len(sub) >= 2:
            annual[y] = round((sub.iloc[-1] / sub.iloc[0] - 1) * 100, 2)

    return {
        "n_trades":  len(df_tr),
        "total_ret": round((nav.iloc[-1] - 1) * 100, 2),
        "sharpe":    round(dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0, 3),
        "mdd":       round(((nav - nav.cummax()) / nav.cummax()).min() * 100, 2),
        "win_rate":  round(df_tr["win"].mean() * 100, 1),
        "avg_hold":  round(df_tr["hold_days"].mean(), 1),
        "pf":        round(abs(wins.mean() / loss.mean()), 2)
                     if len(loss) > 0 and loss.mean() != 0 else 0.0,
        "exit":      df_tr["exit_reason"].value_counts().to_dict(),
        "agent_cnt": df_tr["agent"].value_counts().to_dict(),
        "annual":    annual,
        "nav":       nav,
        "n_rev_sig": 0,
        "n_tr_sig":  0,
    }


# ═══════════════════════════════════════════════════════════════════
# 8. A) Expanding + 정적 200
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("A) Expanding + 정적 200 모델 학습...")
X_all_static, y_all_static, d_all_static = build_rev_dataset(static_set)
print(f"   리버전 전체 샘플: {len(X_all_static):,}  |  양성 비율: {y_all_static.mean():.3f}")
fold_models_A = train_fold_models(WF_FOLDS_EXPANDING, X_all_static, y_all_static, d_all_static)

print("  신호 계산 (A)...")
sigs_A, bt_days, n_rev_A, n_tr_A = build_signals(fold_models_A, universe_by_year=None)
print(f"  리버전: {n_rev_A}건  트렌드: {n_tr_A}건")

print("  시뮬레이션 (A)...")
r_A = run_sim(bt_days, sigs_A)
r_A["n_rev_sig"] = n_rev_A
r_A["n_tr_sig"]  = n_tr_A
print(f"  → {r_A.get('total_ret', 0):+.2f}%  거래={r_A.get('n_trades', 0)}건")


# ═══════════════════════════════════════════════════════════════════
# 8-D. D) Expanding + PIT 200  ← fold_models_A 재사용
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("D) Expanding + PIT 200 신호 계산 (모델은 A 재사용)...")
sigs_D, _, n_rev_D, n_tr_D = build_signals(fold_models_A, universe_by_year=pit_universe_sets)
print(f"  리버전: {n_rev_D}건  트렌드: {n_tr_D}건")

print("  시뮬레이션 (D)...")
r_D = run_sim(bt_days, sigs_D)
r_D["n_rev_sig"] = n_rev_D
r_D["n_tr_sig"]  = n_tr_D
print(f"  → {r_D.get('total_ret', 0):+.2f}%  거래={r_D.get('n_trades', 0)}건")


# ═══════════════════════════════════════════════════════════════════
# 9. B) Rolling 3년 + 정적 200
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("B) Rolling 3년 + 정적 200 모델 학습...")
fold_models_B = train_fold_models(WF_FOLDS_ROLLING, X_all_static, y_all_static, d_all_static)

print("  신호 계산 (B)...")
sigs_B, _, n_rev_B, n_tr_B = build_signals(fold_models_B, universe_by_year=None)
print(f"  리버전: {n_rev_B}건  트렌드: {n_tr_B}건")

print("  시뮬레이션 (B)...")
r_B = run_sim(bt_days, sigs_B)
r_B["n_rev_sig"] = n_rev_B
r_B["n_tr_sig"]  = n_tr_B
print(f"  → {r_B.get('total_ret', 0):+.2f}%  거래={r_B.get('n_trades', 0)}건")


# ═══════════════════════════════════════════════════════════════════
# 10. C) Rolling 3년 + PIT 200
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("C) Rolling 3년 + PIT 200 모델 학습...")

# PIT fold별로 해당 연도 유니버스로 라벨 생성
# 각 fold의 train 기간에서 해당 PIT 유니버스 종목만 사용
# 단순화: 각 fold의 검증 연도 유니버스로 학습 (train 기간 PIT를 별도 구성하기 복잡하므로
#         train fold 종료 시점의 PIT로 근사)
pit_fold_train_sets = {}
for (tr_s, tr_e, ylbl) in WF_FOLDS_ROLLING:
    # train 종료 시점 = val 시작, 해당 연도 PIT 유니버스 사용
    pit_fold_train_sets[ylbl] = pit_universe_sets.get(ylbl, set(static_200))

# PIT 전체 유니버스 합집합으로 데이터셋 구축 후 학습 시 마스킹
pit_all_tickers = all_pit
X_all_pit, y_all_pit, d_all_pit = build_rev_dataset(pit_all_tickers)
print(f"   리버전 전체 샘플 (PIT 풀): {len(X_all_pit):,}  |  양성 비율: {y_all_pit.mean():.3f}")

# Rolling fold 학습 — fold별로 해당 PIT 유니버스 종목만 마스킹
# 실용적으로 전체 PIT 풀로 학습 (fold마다 유니버스가 비슷하고 추가 복잡도 큼)
fold_models_C = train_fold_models(WF_FOLDS_ROLLING, X_all_pit, y_all_pit, d_all_pit)

print("  신호 계산 (C)...")
sigs_C, _, n_rev_C, n_tr_C = build_signals(fold_models_C, universe_by_year=pit_universe_sets)
print(f"  리버전: {n_rev_C}건  트렌드: {n_tr_C}건")

print("  시뮬레이션 (C)...")
r_C = run_sim(bt_days, sigs_C)
r_C["n_rev_sig"] = n_rev_C
r_C["n_tr_sig"]  = n_tr_C
print(f"  → {r_C.get('total_ret', 0):+.2f}%  거래={r_C.get('n_trades', 0)}건")


# ═══════════════════════════════════════════════════════════════════
# 11. 결과 출력
# ═══════════════════════════════════════════════════════════════════
YEARS = ["2023", "2024", "2025", "2026"]
configs = [
    ("A) Expanding+정적200",  r_A),
    ("D) Expanding+PIT200",   r_D),
    ("B) Rolling+정적200",    r_B),
    ("C) Rolling+PIT200",     r_C),
]
W = 20

print(f"\n{'='*87}")
print(f"  4-way 비교: Expanding vs Rolling × 정적 vs PIT")
print(f"  검증 구간: {BT_START} ~ {BT_END}")
print(f"{'='*87}")

# 연도별 수익률
print(f"\n[연도별 수익률]")
hdr = f"  {'연도':>5}" + "".join(f"  {lbl:>{W}}" for lbl, _ in configs)
print(hdr)
print("  " + "-" * (7 + (W+2)*4))
for y in YEARS:
    row = f"  {y:>5}"
    for _, r in configs:
        v = r.get("annual", {}).get(y)
        cell = f"{v:+.2f}%" if v is not None else "   N/A"
        row += f"  {cell:>{W}}"
    print(row)

# 전체 지표
print(f"\n[전체 지표]")
metrics = [
    ("총수익률",  "total_ret", lambda v: f"{v:+.2f}%"),
    ("샤프",      "sharpe",    lambda v: f"{v:.3f}"),
    ("MDD",       "mdd",       lambda v: f"{v:.2f}%"),
    ("거래수",    "n_trades",  lambda v: f"{v}건"),
    ("승률",      "win_rate",  lambda v: f"{v:.1f}%"),
    ("손익비",    "pf",        lambda v: f"{v:.2f}"),
]
hdr = f"  {'항목':>10}" + "".join(f"  {lbl:>{W}}" for lbl, _ in configs)
print(hdr)
print("  " + "-" * (12 + (W+2)*4))
for label, key, fmt_fn in metrics:
    row = f"  {label:>10}"
    for _, r in configs:
        v = r.get(key, 0)
        row += f"  {fmt_fn(v):>{W}}"
    print(row)

# 에이전트별
print(f"\n[에이전트별 거래 수]")
hdr = f"  {'에이전트':>10}" + "".join(f"  {lbl:>{W}}" for lbl, _ in configs)
print(hdr)
print("  " + "-" * (12 + (W+2)*4))
for ag in ["reversion", "trend"]:
    row = f"  {ag:>10}"
    for _, r in configs:
        cnt   = r.get("agent_cnt", {}).get(ag, 0)
        total = sum(r.get("agent_cnt", {}).values()) or 1
        cell  = f"{cnt}건 ({cnt/total*100:.0f}%)"
        row  += f"  {cell:>{W}}"
    print(row)

# 신호 수
print(f"\n[신호 수]")
for label, key in [("리버전 신호", "n_rev_sig"), ("트렌드 신호", "n_tr_sig")]:
    row = f"  {label:>12}"
    for _, r in configs:
        row += f"  {r.get(key, 0):>{W}}"
    print(row)

# 진단 섹션
print(f"\n{'='*75}")
print(f"[진단]")
r_a_ret = r_A.get("total_ret", 0)
r_b_ret = r_B.get("total_ret", 0)
r_c_ret = r_C.get("total_ret", 0)
r_d_ret = r_D.get("total_ret", 0)
print(f"  생존편향 제거 효과 Expanding (A→D): {r_d_ret - r_a_ret:+.2f}%p  ({'생존편향 영향 확인됨' if r_d_ret < r_a_ret else '편향 미미'})")
print(f"  생존편향 제거 효과 Rolling  (B→C): {r_c_ret - r_b_ret:+.2f}%p  ({'생존편향 영향 확인됨' if r_c_ret < r_b_ret else '편향 미미'})")
print(f"  학습방식 변경 효과 정적     (A→B): {r_b_ret - r_a_ret:+.2f}%p  (Rolling이 {'유리' if r_b_ret > r_a_ret else '불리'})")
print(f"  학습방식 변경 효과 PIT      (D→C): {r_c_ret - r_d_ret:+.2f}%p  (Rolling이 {'유리' if r_c_ret > r_d_ret else '불리'})")
print(f"  최종 현실적 성과 D (Expanding+PIT): {r_d_ret:+.2f}%")

print(f"\n  PIT 유니버스 상세:")
for year, pit_list in pit_universes.items():
    overlap = len(set(static_200) & set(pit_list))
    print(f"    {year}: {len(pit_list)}개  (정적200과 {overlap}개 겹침, {200-overlap}개 다름)")

print(f"\n  PIT 누적 등장 종목: {len(all_pit)}개 (정적 200 대비 {len(all_pit)-200:+d}개 추가)")
print(f"  한계: 상폐 종목 미포함, 발행주식수 현재 기준 고정")
print(f"='*75")
