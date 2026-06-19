#!/usr/bin/env python3
"""
backtest_momentum.py — momentum (눌림목 반등) 전역 모델 백테스트

기간  : 2026-01-01 ~ 2026-06-19 (학습 미사용 구간)
진입  : momentum 신호 + win_prob >= 0.52 → 익일 시초가 매수
청산  : TP +15% / SL -8% / 10거래일 경과 (시초가)
비용  : 왕복 0.46%
사이징: 자본의 10% 균등 배분, 최대 10종목 동시 보유
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import yfinance as yf

from ml.features import add_features, detect_momentum_rows

# ── 파라미터 ──────────────────────────────────────────────────────────────
BT_START         = "2026-01-01"
BT_END           = "2026-06-19"
WIN_PROB_THRESH  = 0.52
TP_PCT           = 0.15
SL_PCT           = 0.08
MAX_HOLD         = 10
POSITION_SIZE    = 0.10
MAX_POSITIONS    = 10
COST_RT          = 0.0046


def _strip_tz(idx):
    return idx.tz_localize(None) if idx.tzinfo else idx

def _to_ts(d):
    ts = pd.Timestamp(d)
    return ts.tz_localize(None) if ts.tzinfo else ts

def _row(df, date):
    idx = _strip_tz(df.index)
    mask = (idx == date)
    return df.iloc[np.where(mask)[0]] if mask.any() else pd.DataFrame()


# ── 모델 로드 ─────────────────────────────────────────────────────────────
with open("ml/models/_global_momentum.pkl", "rb") as f:
    saved = pickle.load(f)
model = saved["model"]
fc    = saved["metrics"]["feature_cols"]
print(f"모델 로드: {len(fc)}개 피처  OOF AUC={saved['metrics']['auc']:.4f}")

# ── 데이터 다운로드 ───────────────────────────────────────────────────────
print("KOSPI 다운로드...")
kospi_raw = yf.download("^KS11", period="3y", auto_adjust=True, progress=False)
if isinstance(kospi_raw.columns, pd.MultiIndex):
    kospi_raw.columns = kospi_raw.columns.get_level_values(0)
kospi_raw.index = _strip_tz(kospi_raw.index)

print("유니버스 로드...")
from signals.krx_universe import get_krx_backtest_universe
tickers = list(get_krx_backtest_universe(top_n=200).keys())

all_days = _strip_tz(kospi_raw.index)
bt_days  = all_days[(all_days >= BT_START) & (all_days <= BT_END)]

# ── 신호 생성 ─────────────────────────────────────────────────────────────
print("신호 생성 중...")
signals_by_date: dict[pd.Timestamp, list[dict]] = {}

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
        mask      = detect_momentum_rows(sig_range).reindex(sig_range.index).fillna(False)

        for sig_date in sig_range[mask].index:
            row = sig_range.loc[sig_date]
            X   = row[fc].values.reshape(1, -1).astype("float32")
            wp  = float(model.predict_proba(X)[0, 1])
            if wp >= WIN_PROB_THRESH:
                ts = _to_ts(sig_date)
                signals_by_date.setdefault(ts, []).append({
                    "ticker":   ticker,
                    "win_prob": wp,
                    "df":       df,
                })
    except Exception:
        continue

n_signals = sum(len(v) for v in signals_by_date.values())
print(f"신호 수: {n_signals}건  ({len(signals_by_date)}거래일)")

# ── 포트폴리오 시뮬레이션 ─────────────────────────────────────────────────
capital   = 1.0
positions: list[dict] = []
trades:    list[dict] = []
daily_nav: list[tuple] = []

for i, today in enumerate(bt_days):
    next_day = bt_days[i + 1] if i + 1 < len(bt_days) else None

    # ── 청산 ─────────────────────────────────────────────────────────────
    still_open = []
    for pos in positions:
        pos["hold_days"] += 1
        today_row = _row(pos["df"], today)

        if today_row.empty:
            still_open.append(pos)
            continue

        hi, lo, op = (float(today_row[c].iloc[0]) for c in ("High", "Low", "Open"))
        ep    = pos["entry_price"]
        tp_px = ep * (1 + TP_PCT)
        sl_px = ep * (1 - SL_PCT)

        exit_px = exit_rsn = None
        if lo <= sl_px:
            exit_px, exit_rsn = sl_px, "SL"
        elif hi >= tp_px:
            exit_px, exit_rsn = tp_px, "TP"
        elif pos["hold_days"] >= MAX_HOLD:
            exit_px, exit_rsn = op, "TIME"

        if exit_px is not None:
            net = exit_px / ep - 1 - COST_RT
            capital += pos["size"] * (1 + net)
            trades.append({
                "ticker":      pos["ticker"],
                "entry_date":  pos["entry_date"],
                "exit_date":   today,
                "entry_price": ep,
                "exit_price":  exit_px,
                "exit_reason": exit_rsn,
                "net_ret":     net,
                "hold_days":   pos["hold_days"],
                "win":         net > 0,
            })
        else:
            still_open.append(pos)
    positions = still_open

    # ── 진입 ─────────────────────────────────────────────────────────────
    if next_day is not None and today in signals_by_date:
        cands = sorted(signals_by_date[today], key=lambda x: -x["win_prob"])
        for sig in cands:
            if len(positions) >= MAX_POSITIONS:
                break
            if any(p["ticker"] == sig["ticker"] for p in positions):
                continue
            entry_row = _row(sig["df"], next_day)
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
                "entry_date":  next_day,
                "hold_days":   0,
                "size":        size,
                "df":          sig["df"],
            })

    # ── NAV ──────────────────────────────────────────────────────────────
    pos_mkt = 0.0
    for pos in positions:
        tr = _row(pos["df"], today)
        cp = float(tr["Close"].iloc[0]) if not tr.empty else pos["entry_price"]
        pos_mkt += pos["size"] * (cp / pos["entry_price"])
    daily_nav.append((today, capital + pos_mkt))


# ── 결과 집계 ─────────────────────────────────────────────────────────────
if not trades:
    print("\n거래 없음")
else:
    df_tr = pd.DataFrame(trades)
    df_eq = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")

    total_ret = df_eq["nav"].iloc[-1] - 1.0
    dr        = df_eq["nav"].pct_change().dropna()
    sharpe    = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    mdd       = ((df_eq["nav"] - df_eq["nav"].cummax()) / df_eq["nav"].cummax()).min()

    wins   = df_tr[df_tr["net_ret"] > 0]["net_ret"]
    losses = df_tr[df_tr["net_ret"] <= 0]["net_ret"]
    avg_win  = wins.mean()   if len(wins)   > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    pf       = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    print(f"\n{'='*52}")
    print(f"백테스트 결과  {BT_START} ~ {BT_END}")
    print(f"{'='*52}")
    print(f"총 수익률     : {total_ret*100:+.2f}%")
    print(f"샤프 비율     : {sharpe:.3f}")
    print(f"최대 낙폭(MDD): {mdd*100:.2f}%")
    print(f"총 거래 수    : {len(df_tr)}")
    print(f"승률          : {df_tr['win'].mean()*100:.1f}%")
    print(f"평균 수익     : {avg_win*100:+.2f}%")
    print(f"평균 손실     : {avg_loss*100:+.2f}%")
    print(f"손익비        : {pf:.2f}")
    print(f"\n청산 사유:")
    for reason, cnt in df_tr["exit_reason"].value_counts().items():
        print(f"  {reason}: {cnt}건")
    print(f"\n상위 5 거래:")
    for _, r in df_tr.nlargest(5, "net_ret")[["ticker","entry_date","exit_date","net_ret","exit_reason"]].iterrows():
        print(f"  {r['ticker']:12s} {str(r['entry_date'])[:10]} → {str(r['exit_date'])[:10]}  {r['net_ret']*100:+.2f}%  [{r['exit_reason']}]")
    print(f"\n하위 5 거래:")
    for _, r in df_tr.nsmallest(5, "net_ret")[["ticker","entry_date","exit_date","net_ret","exit_reason"]].iterrows():
        print(f"  {r['ticker']:12s} {str(r['entry_date'])[:10]} → {str(r['exit_date'])[:10]}  {r['net_ret']*100:+.2f}%  [{r['exit_reason']}]")

    monthly = df_eq["nav"].resample("ME").last().pct_change().dropna()
    if not monthly.empty:
        print(f"\n월별 수익률:")
        for d, r in monthly.items():
            print(f"  {str(d)[:7]}: {r*100:+.2f}%")
