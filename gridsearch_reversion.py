#!/usr/bin/env python3
"""
gridsearch_reversion.py — reversion 모델 파라미터 그리드 서치

데이터/신호는 한 번만 다운로드하고 100개 조합을 재사용.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import pickle
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
from collections import Counter

from ml.features import add_features, detect_reversion_rows

# ── 고정 파라미터 ──────────────────────────────────────────────────────────
BT_START        = "2026-01-01"
BT_END          = "2026-06-19"
WIN_PROB_THRESH = 0.52
POSITION_SIZE   = 0.10
MAX_POSITIONS   = 10
COST_RT         = 0.0046

# ── 그리드 ────────────────────────────────────────────────────────────────
TP_LIST   = [0.08, 0.10, 0.12, 0.15, 0.18]
SL_LIST   = [0.04, 0.05, 0.06, 0.07, 0.08]
HOLD_LIST = [5, 7, 10, 14]

# ── 필터 조건 ─────────────────────────────────────────────────────────────
MIN_SHARPE  =  1.0
MAX_MDD     = -0.15
MIN_TRADES  =  50
MIN_PF      =  1.5


def _strip_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_localize(None) if idx.tzinfo else idx


def _to_ts(d) -> pd.Timestamp:
    ts = pd.Timestamp(d)
    return ts.tz_localize(None) if ts.tzinfo else ts


def _row(df: pd.DataFrame, date: pd.Timestamp):
    idx = _strip_tz(df.index)
    pos = np.where(idx == date)[0]
    return df.iloc[pos] if len(pos) else pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드 / 신호 생성
# ════════════════════════════════════════════════════════════════════════════
print("모델 로드...")
with open("ml/models/_global_reversion.pkl", "rb") as f:
    saved = pickle.load(f)
model = saved["model"]
fc    = saved["metrics"]["feature_cols"]
print(f"  {len(fc)}개 피처  OOF AUC={saved['metrics']['auc']:.4f}")

print("KOSPI 다운로드...")
kospi_raw = yf.download("^KS11", period="3y", auto_adjust=True, progress=False)
if isinstance(kospi_raw.columns, pd.MultiIndex):
    kospi_raw.columns = kospi_raw.columns.get_level_values(0)
kospi_raw.index = _strip_tz(kospi_raw.index)

print("유니버스 로드...")
from signals.krx_universe import get_krx_backtest_universe
tickers = list(get_krx_backtest_universe(top_n=200).keys())

all_dfs: dict[str, pd.DataFrame] = {}  # {ticker: ohlcv}
# signals_by_date[date] = [{ticker, win_prob}, ...]
signals_by_date: dict[pd.Timestamp, list[dict]] = {}

print("신호 생성 중...")
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

        sig_range = df_feat[
            (df_feat.index >= BT_START) & (df_feat.index < BT_END)
        ]
        rev_mask = detect_reversion_rows(sig_range).reindex(sig_range.index).fillna(False)

        for sig_date in sig_range[rev_mask].index:
            row = sig_range.loc[sig_date]
            X = row[fc].values.reshape(1, -1).astype("float32")
            wp = float(model.predict_proba(X)[0, 1])
            if wp >= WIN_PROB_THRESH:
                ts = _to_ts(sig_date)
                signals_by_date.setdefault(ts, []).append({
                    "ticker":   ticker,
                    "win_prob": wp,
                })
                all_dfs[ticker] = df  # OHLCV 저장 (없으면 추가)
    except Exception:
        continue

n_sig = sum(len(v) for v in signals_by_date.values())
print(f"  신호: {n_sig}건 ({len(signals_by_date)}거래일)  종목 OHLCV: {len(all_dfs)}개")

# 거래일 목록
all_days = _strip_tz(kospi_raw.index)
bt_days  = all_days[(all_days >= BT_START) & (all_days <= BT_END)]


# ════════════════════════════════════════════════════════════════════════════
# 2. 시뮬레이션 함수
# ════════════════════════════════════════════════════════════════════════════
def run_sim(tp: float, sl: float, hold: int) -> dict:
    capital   = 1.0
    positions: list[dict] = []
    trades:    list[dict] = []
    daily_nav: list[float] = []

    for i, today in enumerate(bt_days):
        next_day = bt_days[i + 1] if i + 1 < len(bt_days) else None

        # ── 청산 체크 ────────────────────────────────────────────────────
        still_open = []
        for pos in positions:
            pos["hold_days"] += 1
            df_pos   = all_dfs.get(pos["ticker"])
            today_row = _row(df_pos, today) if df_pos is not None else pd.DataFrame()

            if today_row.empty:
                still_open.append(pos)
                continue

            hi, lo, op = (float(today_row[c].iloc[0]) for c in ("High", "Low", "Open"))
            ep     = pos["entry_price"]
            tp_px  = ep * (1 + tp)
            sl_px  = ep * (1 - sl)

            exit_px = exit_rsn = None
            if lo <= sl_px:
                exit_px, exit_rsn = sl_px, "SL"
            elif hi >= tp_px:
                exit_px, exit_rsn = tp_px, "TP"
            elif pos["hold_days"] >= hold:
                exit_px, exit_rsn = op, "TIME"

            if exit_px is not None:
                gross = exit_px / ep - 1
                net   = gross - COST_RT
                capital += pos["size"] * (1 + net)
                trades.append({"net_ret": net, "exit_reason": exit_rsn, "win": net > 0})
            else:
                still_open.append(pos)

        positions = still_open

        # ── 신규 진입 ────────────────────────────────────────────────────
        if next_day is not None and today in signals_by_date:
            cands = sorted(signals_by_date[today], key=lambda x: -x["win_prob"])
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

        # ── NAV 기록 ─────────────────────────────────────────────────────
        pos_mkt = 0.0
        for pos in positions:
            df_pos = all_dfs.get(pos["ticker"])
            if df_pos is None:
                pos_mkt += pos["size"]
                continue
            tr = _row(df_pos, today)
            cp = float(tr["Close"].iloc[0]) if not tr.empty else pos["entry_price"]
            pos_mkt += pos["size"] * (cp / pos["entry_price"])

        daily_nav.append(capital + pos_mkt)

    if not trades or len(daily_nav) < 2:
        return {}

    df_tr = pd.DataFrame(trades)
    nav   = pd.Series(daily_nav)
    dr    = nav.pct_change().dropna()

    total_ret = nav.iloc[-1] - 1.0
    sharpe    = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    mdd       = ((nav - nav.cummax()) / nav.cummax()).min()
    n_trades  = len(df_tr)
    win_rate  = df_tr["win"].mean()
    wins      = df_tr[df_tr["net_ret"] > 0]["net_ret"]
    losses    = df_tr[df_tr["net_ret"] <= 0]["net_ret"]
    avg_win   = wins.mean()   if len(wins)   > 0 else 0.0
    avg_loss  = losses.mean() if len(losses) > 0 else 0.0
    pf        = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    return {
        "tp": tp, "sl": sl, "hold": hold,
        "total_ret": round(total_ret * 100, 2),
        "sharpe":    round(sharpe, 3),
        "mdd":       round(mdd * 100, 2),
        "n_trades":  n_trades,
        "win_rate":  round(win_rate * 100, 1),
        "avg_win":   round(avg_win * 100, 2),
        "avg_loss":  round(avg_loss * 100, 2),
        "pf":        round(pf, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. 그리드 서치 실행
# ════════════════════════════════════════════════════════════════════════════
combos = list(itertools.product(TP_LIST, SL_LIST, HOLD_LIST))
print(f"\n그리드 서치 시작: {len(combos)}개 조합...")

all_results = []
for idx, (tp, sl, hold) in enumerate(combos):
    res = run_sim(tp, sl, hold)
    if res:
        all_results.append(res)
    if (idx + 1) % 20 == 0:
        print(f"  진행: {idx + 1} / {len(combos)}")

print(f"완료. 유효 결과: {len(all_results)}건")


# ════════════════════════════════════════════════════════════════════════════
# 4. 필터 + 출력
# ════════════════════════════════════════════════════════════════════════════
valid = [
    r for r in all_results
    if r["sharpe"]   >= MIN_SHARPE
    and r["mdd"]     >= MAX_MDD * 100
    and r["n_trades"] >= MIN_TRADES
    and r["pf"]      >= MIN_PF
]
valid.sort(key=lambda x: -x["sharpe"])

print(f"\n{'='*70}")
print(f"유효 조합: {len(valid)} / {len(all_results)}개  (샤프≥{MIN_SHARPE}, MDD≥{MAX_MDD*100}%, 거래≥{MIN_TRADES}, 손익비≥{MIN_PF})")
print(f"{'='*70}")

if valid:
    # 유효 조합 전체 (샤프 내림차순)
    print(f"\n{'TP':>5} {'SL':>5} {'Hold':>5} | {'수익률':>8} {'샤프':>7} {'MDD':>8} {'거래':>5} {'승률':>7} {'손익비':>7}")
    print("-" * 70)
    for r in valid:
        print(f"{r['tp']*100:>4.0f}% {r['sl']*100:>4.0f}% {r['hold']:>5}일 | "
              f"{r['total_ret']:>7.2f}% {r['sharpe']:>7.3f} {r['mdd']:>7.2f}% "
              f"{r['n_trades']:>5} {r['win_rate']:>6.1f}% {r['pf']:>7.2f}")

    # 상위 5개 상세
    print(f"\n{'='*70}")
    print("상위 5개 상세 결과")
    print(f"{'='*70}")
    for i, r in enumerate(valid[:5], 1):
        print(f"\n[{i}위] TP={r['tp']*100:.0f}%  SL={r['sl']*100:.0f}%  보유={r['hold']}일")
        print(f"  총 수익률 : {r['total_ret']:+.2f}%")
        print(f"  샤프 비율 : {r['sharpe']:.3f}")
        print(f"  MDD       : {r['mdd']:.2f}%")
        print(f"  총 거래   : {r['n_trades']}건")
        print(f"  승률      : {r['win_rate']:.1f}%")
        print(f"  평균 수익 : {r['avg_win']:+.2f}%")
        print(f"  평균 손실 : {r['avg_loss']:+.2f}%")
        print(f"  손익비    : {r['pf']:.2f}")

    # 파라미터 안정성
    print(f"\n{'='*70}")
    print("파라미터 등장 빈도 (안정성)")
    print(f"{'='*70}")
    tp_cnt   = Counter(r["tp"]   for r in valid)
    sl_cnt   = Counter(r["sl"]   for r in valid)
    hold_cnt = Counter(r["hold"] for r in valid)
    print("TP   :", {f"{k*100:.0f}%": v for k, v in sorted(tp_cnt.items(), key=lambda x: -x[1])})
    print("SL   :", {f"{k*100:.0f}%": v for k, v in sorted(sl_cnt.items(), key=lambda x: -x[1])})
    print("Hold :", {f"{k}일": v        for k, v in sorted(hold_cnt.items(), key=lambda x: -x[1])})
else:
    print("\n유효 조합 없음 — 필터 기준을 완화해보세요.")
    # 필터 미적용 상위 10개
    all_results.sort(key=lambda x: -x["sharpe"])
    print("\n전체 결과 상위 10개 (필터 없음):")
    print(f"{'TP':>5} {'SL':>5} {'Hold':>5} | {'수익률':>8} {'샤프':>7} {'MDD':>8} {'거래':>5} {'승률':>7} {'손익비':>7}")
    print("-" * 70)
    for r in all_results[:10]:
        print(f"{r['tp']*100:>4.0f}% {r['sl']*100:>4.0f}% {r['hold']:>5}일 | "
              f"{r['total_ret']:>7.2f}% {r['sharpe']:>7.3f} {r['mdd']:>7.2f}% "
              f"{r['n_trades']:>5} {r['win_rate']:>6.1f}% {r['pf']:>7.2f}")
