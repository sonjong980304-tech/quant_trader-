#!/usr/bin/env python3
"""
winprob_compare.py — win_prob 기준값별 백테스트 비교

TP=15%  SL=8%  보유=10일 (그리드서치 최적값)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import yfinance as yf

from ml.features import add_features, detect_reversion_rows

# ── 파라미터 ──────────────────────────────────────────────────────────────
BT_START      = "2026-01-01"
BT_END        = "2026-06-19"
TP            = 0.15
SL            = 0.08
MAX_HOLD      = 10
POSITION_SIZE = 0.10
MAX_POSITIONS = 10
COST_RT       = 0.0046
THRESHOLDS    = [0.52, 0.55, 0.58, 0.60]


def _strip_tz(idx):
    return idx.tz_localize(None) if idx.tzinfo else idx


def _row(df, date):
    idx = _strip_tz(df.index)
    pos = np.where(idx == date)[0]
    return df.iloc[pos] if len(pos) else pd.DataFrame()


# ── 모델 로드 ─────────────────────────────────────────────────────────────
print("모델 로드...")
with open("ml/models/_global_reversion.pkl", "rb") as f:
    saved = pickle.load(f)
model = saved["model"]
fc    = saved["metrics"]["feature_cols"]

# ── 데이터 다운로드 ───────────────────────────────────────────────────────
print("KOSPI 다운로드...")
kospi_raw = yf.download("^KS11", period="3y", auto_adjust=True, progress=False)
if isinstance(kospi_raw.columns, pd.MultiIndex):
    kospi_raw.columns = kospi_raw.columns.get_level_values(0)
kospi_raw.index = _strip_tz(kospi_raw.index)

print("유니버스 로드...")
from signals.krx_universe import get_krx_backtest_universe
tickers = list(get_krx_backtest_universe(top_n=200).keys())

# ── 신호 생성 (win_prob 저장) ─────────────────────────────────────────────
print("신호 생성 중...")
all_dfs: dict[str, pd.DataFrame] = {}
# raw_signals[date] = [{ticker, win_prob}, ...]
raw_signals: dict[pd.Timestamp, list[dict]] = {}

for ticker in tickers:
    try:
        df = yf.download(ticker, period="3y", auto_adjust=True, progress=False)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = _strip_tz(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        df_feat = add_features(df, kospi_df=kospi_raw)
        df_feat.index = _strip_tz(df_feat.index)
        df_feat = df_feat.dropna(subset=fc)

        sig_range = df_feat[(df_feat.index >= BT_START) & (df_feat.index < BT_END)]
        rev_mask  = detect_reversion_rows(sig_range).reindex(sig_range.index).fillna(False)

        for sig_date in sig_range[rev_mask].index:
            row = sig_range.loc[sig_date]
            X   = row[fc].values.reshape(1, -1).astype("float32")
            wp  = float(model.predict_proba(X)[0, 1])
            ts  = pd.Timestamp(sig_date).tz_localize(None) if pd.Timestamp(sig_date).tzinfo else pd.Timestamp(sig_date)
            raw_signals.setdefault(ts, []).append({"ticker": ticker, "win_prob": wp})
            all_dfs[ticker] = df
    except Exception:
        continue

total_raw = sum(len(v) for v in raw_signals.values())
print(f"  전체 reversion 신호: {total_raw}건")
for thr in THRESHOLDS:
    n = sum(1 for sigs in raw_signals.values() for s in sigs if s["win_prob"] >= thr)
    print(f"  win_prob >= {thr}: {n}건")

# 거래일 목록
all_days = _strip_tz(kospi_raw.index)
bt_days  = all_days[(all_days >= BT_START) & (all_days <= BT_END)]


# ── 시뮬레이션 함수 ───────────────────────────────────────────────────────
def run_sim(threshold: float) -> dict:
    capital   = 1.0
    positions: list[dict] = []
    trades:    list[dict] = []
    daily_nav: list[float] = []

    for i, today in enumerate(bt_days):
        next_day = bt_days[i + 1] if i + 1 < len(bt_days) else None

        # 청산
        still_open = []
        for pos in positions:
            pos["hold_days"] += 1
            df_pos    = all_dfs.get(pos["ticker"])
            today_row = _row(df_pos, today) if df_pos is not None else pd.DataFrame()

            if today_row.empty:
                still_open.append(pos)
                continue

            hi, lo, op = (float(today_row[c].iloc[0]) for c in ("High", "Low", "Open"))
            ep    = pos["entry_price"]
            exit_px = exit_rsn = None
            if lo <= ep * (1 - SL):
                exit_px, exit_rsn = ep * (1 - SL), "SL"
            elif hi >= ep * (1 + TP):
                exit_px, exit_rsn = ep * (1 + TP), "TP"
            elif pos["hold_days"] >= MAX_HOLD:
                exit_px, exit_rsn = op, "TIME"

            if exit_px is not None:
                net = exit_px / ep - 1 - COST_RT
                capital += pos["size"] * (1 + net)
                trades.append({"net_ret": net, "win": net > 0})
            else:
                still_open.append(pos)
        positions = still_open

        # 진입
        if next_day is not None and today in raw_signals:
            cands = sorted(
                [s for s in raw_signals[today] if s["win_prob"] >= threshold],
                key=lambda x: -x["win_prob"],
            )
            for sig in cands:
                if len(positions) >= MAX_POSITIONS:
                    break
                if any(p["ticker"] == sig["ticker"] for p in positions):
                    continue
                df_pos = all_dfs.get(sig["ticker"])
                if df_pos is None:
                    continue
                entry_row = _row(df_pos, next_day)
                if entry_row.empty:
                    continue
                ep = float(entry_row["Open"].iloc[0])
                if ep <= 0:
                    continue
                size = capital * POSITION_SIZE
                capital -= size * (1 + COST_RT / 2)
                positions.append({
                    "ticker":      sig["ticker"],
                    "entry_price": ep,
                    "hold_days":   0,
                    "size":        size,
                })

        # NAV
        pos_mkt = 0.0
        for pos in positions:
            df_pos = all_dfs.get(pos["ticker"])
            tr = _row(df_pos, today) if df_pos is not None else pd.DataFrame()
            cp = float(tr["Close"].iloc[0]) if not tr.empty else pos["entry_price"]
            pos_mkt += pos["size"] * (cp / pos["entry_price"])
        daily_nav.append(capital + pos_mkt)

    if not trades:
        return {"threshold": threshold, "n_trades": 0}

    df_tr = pd.DataFrame(trades)
    nav   = pd.Series(daily_nav)
    dr    = nav.pct_change().dropna()

    wins   = df_tr[df_tr["net_ret"] > 0]["net_ret"]
    losses = df_tr[df_tr["net_ret"] <= 0]["net_ret"]
    avg_win  = wins.mean()   if len(wins)   > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    pf = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    return {
        "threshold":  threshold,
        "n_trades":   len(df_tr),
        "total_ret":  round((nav.iloc[-1] - 1) * 100, 2),
        "sharpe":     round(dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0, 3),
        "mdd":        round(((nav - nav.cummax()) / nav.cummax()).min() * 100, 2),
        "win_rate":   round(df_tr["win"].mean() * 100, 1),
        "avg_win":    round(avg_win * 100, 2),
        "avg_loss":   round(avg_loss * 100, 2),
        "pf":         round(pf, 2),
    }


# ── 결과 출력 ─────────────────────────────────────────────────────────────
print("\nwin_prob 기준값별 시뮬레이션 중...")
results = [run_sim(thr) for thr in THRESHOLDS]

print(f"\n{'='*72}")
print(f"win_prob 비교  (TP={TP*100:.0f}%  SL={SL*100:.0f}%  보유={MAX_HOLD}일)")
print(f"{'='*72}")
print(f"{'임계값':>8} {'거래수':>6} {'수익률':>8} {'샤프':>7} {'MDD':>8} {'승률':>7} {'평균수익':>8} {'평균손실':>8} {'손익비':>7}")
print("-" * 72)
for r in results:
    if r["n_trades"] == 0:
        print(f"  >= {r['threshold']:.2f}  거래 없음")
        continue
    print(f"  >= {r['threshold']:.2f}  {r['n_trades']:>5}건  "
          f"{r['total_ret']:>7.2f}%  {r['sharpe']:>6.3f}  {r['mdd']:>7.2f}%  "
          f"{r['win_rate']:>6.1f}%  {r['avg_win']:>+7.2f}%  {r['avg_loss']:>+7.2f}%  {r['pf']:>6.2f}")
