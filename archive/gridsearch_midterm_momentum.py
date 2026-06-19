#!/usr/bin/env python3
"""
gridsearch_midterm_momentum.py — 중기 모멘텀 에이전트 라벨 파라미터 그리드 서치

TP [0.12,0.15,0.18,0.20] × SL [0.06,0.07,0.08] × hold [15,20,25,30] = 48개 조합
데이터/피처: 한 번만 계산, sector_momentum_5d 합산 후 계산
백테스트: 2026-01-01 ~ 2026-06-19, win_prob >= 0.52
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import itertools, csv, os
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
TP_LIST   = [0.12, 0.15, 0.18, 0.20]
SL_LIST   = [0.06, 0.07, 0.08]
HOLD_LIST = [15, 20, 25, 30]

WF_FOLDS = [
    ("2023-01-01", "2024-01-01", "2025-01-01"),
    ("2023-01-01", "2025-01-01", "2026-01-01"),
    ("2023-01-01", "2026-01-01", "2027-01-01"),
]

BT_START        = "2026-01-01"
BT_END          = "2026-06-19"
WIN_PROB_THRESH = 0.52
POSITION_SIZE   = 0.10
MAX_POSITIONS   = 10
COST_RT         = 0.0046

# 필터 기준
MIN_SHARPE = 1.0
MAX_MDD    = -0.15
MIN_TRADES = 30
MIN_PF     = 1.5

FC = FEATURE_COLS_MIDTERM_MOMENTUM

REVERSION_BASELINE = {
    "total_ret": 20.86, "sharpe": 1.976, "mdd": -7.16,
    "n_trades": 84, "win_rate": 59.5, "pf": 1.59,
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

# ── 섹터 매핑 로드 ────────────────────────────────────────────────────────
SECTOR_MAP_PATH = "ml/models/sector_map.csv"
sector_map: dict[str, str] = {}
if os.path.exists(SECTOR_MAP_PATH):
    with open(SECTOR_MAP_PATH, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) == 2:
                sector_map[row[0]] = row[1]
    print(f"섹터 매핑 로드: {len(sector_map)}개")
else:
    print("섹터 매핑 없음 — sector_momentum_5d 조건 skip")

# ── sector_momentum_5d 계산 (합산 후) ────────────────────────────────────
if sector_map:
    print("sector_momentum_5d 계산 중...")
    # 전체 ret_5d + 날짜 + 섹터 합산
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
        # 날짜별 섹터 평균 ret_5d
        sect_avg = (
            all_ret[all_ret["_sector"] != "unknown"]
            .groupby(["_date", "_sector"])["ret_5d"]
            .mean()
        )
        # 각 종목에 sector_momentum_5d 덮어씀
        for ticker, df_f in feat_data.items():
            sector = sector_map.get(ticker, "unknown")
            if sector == "unknown":
                continue
            try:
                dates = pd.to_datetime(df_f.index)
                vals  = sect_avg.get(sector, pd.Series(dtype=float))
                if hasattr(vals, "unstack"):
                    pass
                # (date, sector) MultiIndex → date별 조회
                if isinstance(sect_avg.index, pd.MultiIndex):
                    sec_series = sect_avg.xs(sector, level="_sector") if sector in sect_avg.index.get_level_values("_sector") else pd.Series(dtype=float)
                else:
                    sec_series = pd.Series(dtype=float)
                aligned = sec_series.reindex(dates.normalize()).values
                df_f["sector_momentum_5d"] = aligned
            except Exception:
                pass

    print("sector_momentum_5d 계산 완료")

# ── dropna (FC 기준) 후 학습용 feat_data_clean 생성 ──────────────────────
feat_data_clean: dict[str, pd.DataFrame] = {}
for ticker, df_f in feat_data.items():
    sub = df_f.dropna(subset=FC)
    if len(sub) >= 50:
        feat_data_clean[ticker] = sub

print(f"학습 가능 종목: {len(feat_data_clean)}개")

all_days = _strip_tz(kospi_raw.index)
bt_days  = all_days[(all_days >= BT_START) & (all_days <= BT_END)]


# ════════════════════════════════════════════════════════════════════════════
# 2. 학습 + 백테스트 함수
# ════════════════════════════════════════════════════════════════════════════
def run_one(tp: float, sl: float, hold: int) -> dict:
    # ── 라벨 재생성 ──────────────────────────────────────────────────────
    all_rows = []
    for ticker, df_f in feat_data_clean.items():
        try:
            labels, future_ret = _triple_barrier_pnl(
                df_f, tp_pct=tp, sl_pct=sl, max_holding_days=hold
            )
            tmp = df_f.copy()
            tmp["_label"]  = labels
            tmp["_future"] = future_ret
            tmp["_ticker"] = ticker
            tmp["_date"]   = df_f.index
            tmp = tmp.dropna(subset=["_label","_future"]).iloc[:-hold]
            if len(tmp) >= 5:
                all_rows.append(tmp)
        except Exception:
            pass

    if not all_rows:
        return {"tp": tp, "sl": sl, "hold": hold, "oof_auc": 0.0, "n_trades": 0}

    combined = pd.concat(all_rows).sort_values("_date").reset_index(drop=True)
    X     = combined[FC].values.astype("float32")
    y     = combined["_label"].values.astype(int)
    dates = pd.to_datetime(combined["_date"].values)

    # ── Walk-forward CV ──────────────────────────────────────────────────
    oof_proba = np.full(len(X), np.nan)
    fold_aucs = []

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
            fa       = roc_auc_score(y_vl, proba)
            vl_year  = pd.Timestamp(vl_end).year - 1
            fold_aucs.append((f"valid {vl_year}", round(fa, 4)))
        except Exception:
            pass

    valid_mask = ~np.isnan(oof_proba)
    try:
        oof_auc = roc_auc_score(y[valid_mask], oof_proba[valid_mask]) if valid_mask.any() else 0.0
    except Exception:
        oof_auc = 0.0

    # ── 최종 모델 학습 ───────────────────────────────────────────────────
    n          = len(X)
    calib_size = max(20, int(n * 0.20))
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

    fi = sorted(zip(FC, final.feature_importances_), key=lambda x: -x[1])

    # OOF win_prob 분포
    op_cal = cal_model.predict_proba(X[valid_mask])[:, 1]
    wp_stats = {
        "min":      round(float(op_cal.min()),  4),
        "mean":     round(float(op_cal.mean()), 4),
        "max":      round(float(op_cal.max()),  4),
        "above_52": round(float((op_cal >= 0.52).mean()), 4),
    }

    # ── SHAP 피처 중요도 (가능하면) ─────────────────────────────────────
    shap_fi = None
    try:
        import shap
        explainer  = shap.TreeExplainer(final)
        shap_vals  = explainer.shap_values(X[:min(5000, len(X))])
        shap_mean  = np.abs(shap_vals).mean(axis=0)
        shap_fi    = sorted(zip(FC, shap_mean.tolist()), key=lambda x: -x[1])
    except Exception:
        pass

    # ── 백테스트 신호 생성 ───────────────────────────────────────────────
    signals_by_date: dict[pd.Timestamp, list[dict]] = {}
    for ticker, df_f in feat_data_clean.items():
        try:
            sig_range = df_f[(df_f.index >= BT_START) & (df_f.index < BT_END)]
            mask      = detect_midterm_momentum_rows(sig_range).reindex(sig_range.index).fillna(False)
            for sig_date in sig_range[mask].index:
                row = sig_range.loc[sig_date]
                Xp  = row[FC].values.reshape(1, -1).astype("float32")
                wp  = float(cal_model.predict_proba(Xp)[0, 1])
                if wp >= WIN_PROB_THRESH:
                    ts = _to_ts(sig_date)
                    signals_by_date.setdefault(ts, []).append({
                        "ticker": ticker, "win_prob": wp,
                    })
        except Exception:
            pass

    # ── 시뮬레이션 ──────────────────────────────────────────────────────
    capital = 1.0
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
            if lo <= ep * (1 - sl):
                exit_px, exit_rsn = ep * (1 - sl), "SL"
            elif hi >= ep * (1 + tp):
                exit_px, exit_rsn = ep * (1 + tp), "TP"
            elif pos["hold_days"] >= hold:
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

    base = {"tp": tp, "sl": sl, "hold": hold, "oof_auc": round(oof_auc, 4),
            "fold_aucs": fold_aucs, "n_samples": n,
            "wp_stats": wp_stats, "fi": fi, "shap_fi": shap_fi}

    if not trades or len(daily_nav) < 2:
        return {**base, "n_trades": 0}

    df_tr = pd.DataFrame(trades)
    nav   = pd.Series(daily_nav)
    dr    = nav.pct_change().dropna()
    wins  = df_tr[df_tr["net_ret"] > 0]["net_ret"]
    losses= df_tr[df_tr["net_ret"] <= 0]["net_ret"]

    return {
        **base,
        "n_trades":  len(df_tr),
        "total_ret": round((nav.iloc[-1] - 1) * 100, 2),
        "sharpe":    round(dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0, 3),
        "mdd":       round(((nav - nav.cummax()) / nav.cummax()).min() * 100, 2),
        "win_rate":  round(df_tr["win"].mean() * 100, 1),
        "avg_win":   round(wins.mean()   * 100, 2) if len(wins)   > 0 else 0.0,
        "avg_loss":  round(losses.mean() * 100, 2) if len(losses) > 0 else 0.0,
        "pf":        round(abs(wins.mean() / losses.mean()), 2)
                     if len(losses) > 0 and losses.mean() != 0 else 0.0,
        "exit_counts": df_tr["exit_reason"].value_counts().to_dict(),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. 그리드 서치 실행
# ════════════════════════════════════════════════════════════════════════════
combos = list(itertools.product(TP_LIST, SL_LIST, HOLD_LIST))
print(f"\n그리드 서치 시작: {len(combos)}개 조합...")

results = []
for i, (tp, sl, hold) in enumerate(combos, 1):
    print(f"  [{i:2d}/{len(combos)}] TP={tp*100:.0f}%  SL={sl*100:.0f}%  hold={hold}d ...",
          end=" ", flush=True)
    r = run_one(tp, sl, hold)
    results.append(r)
    if r.get("n_trades", 0) > 0:
        print(f"AUC={r['oof_auc']:.4f}  거래={r['n_trades']}건  샤프={r.get('sharpe', 0):.3f}")
    else:
        print(f"AUC={r['oof_auc']:.4f}  거래없음")


# ════════════════════════════════════════════════════════════════════════════
# 4. 결과 출력
# ════════════════════════════════════════════════════════════════════════════
valid = [
    r for r in results
    if r.get("n_trades", 0) >= MIN_TRADES
    and r.get("sharpe", -999) >= MIN_SHARPE
    and r.get("mdd", -999) >= MAX_MDD * 100
    and r.get("pf", 0) >= MIN_PF
]
valid.sort(key=lambda x: -x.get("sharpe", -999))

print(f"\n{'='*88}")
print(f"중기 모멘텀 그리드 서치  (win_prob>={WIN_PROB_THRESH}  KR {BT_START}~{BT_END})")
print(f"유효 조합: {len(valid)} / {len(results)}개"
      f"  (샤프>={MIN_SHARPE}  MDD>={MAX_MDD*100}%  거래>={MIN_TRADES}건  손익비>={MIN_PF})")
print(f"{'='*88}")

if valid:
    print(f"\n{'TP':>5} {'SL':>5} {'hold':>5} {'OOF AUC':>8} {'수익률':>8} "
          f"{'샤프':>7} {'MDD':>8} {'거래':>5} {'승률':>7} {'손익비':>7}")
    print("-" * 88)
    for r in valid:
        print(f"{r['tp']*100:>4.0f}%  {r['sl']*100:>4.0f}%  {r['hold']:>4}d"
              f"  {r['oof_auc']:>8.4f}"
              f"  {r['total_ret']:>7.2f}%  {r['sharpe']:>6.3f}  {r['mdd']:>7.2f}%"
              f"  {r['n_trades']:>5}  {r['win_rate']:>6.1f}%  {r['pf']:>6.2f}")

    # 상위 5개 상세
    print(f"\n{'='*60}")
    print("상위 5개 조합 상세")
    print(f"{'='*60}")
    for rank, r in enumerate(valid[:5], 1):
        print(f"\n[{rank}위] TP={r['tp']*100:.0f}%  SL={r['sl']*100:.0f}%  보유={r['hold']}일")
        print(f"  OOF AUC     : {r['oof_auc']:.4f}")
        for fd, fa in r.get("fold_aucs", []):
            print(f"    {fd}: {fa:.4f}")
        print(f"  총 수익률   : {r['total_ret']:+.2f}%")
        print(f"  샤프 비율   : {r['sharpe']:.3f}")
        print(f"  MDD         : {r['mdd']:.2f}%")
        print(f"  거래 수     : {r['n_trades']}건")
        print(f"  승률        : {r['win_rate']:.1f}%")
        print(f"  평균 수익   : {r['avg_win']:+.2f}%")
        print(f"  평균 손실   : {r['avg_loss']:+.2f}%")
        print(f"  손익비      : {r['pf']:.2f}")
        if r.get("exit_counts"):
            print(f"  청산 사유   : {r['exit_counts']}")

    # 최적 조합 피처 중요도
    best = valid[0]
    print(f"\n{'='*60}")
    print(f"최적 조합 피처 중요도: TP={best['tp']*100:.0f}% SL={best['sl']*100:.0f}% hold={best['hold']}d")
    print(f"{'='*60}")
    fi_to_show = best.get("shap_fi") or best.get("fi", [])
    fi_label   = "SHAP" if best.get("shap_fi") else "Feature Importance"
    print(f"  ({fi_label})")
    for feat, imp in fi_to_show:
        print(f"    {feat:30s}  {imp:.4f}")

    ws = best.get("wp_stats", {})
    print(f"\n  win_prob 분포 (OOF, 캘리브레이션 후)")
    print(f"    min={ws.get('min',0):.4f}  mean={ws.get('mean',0):.4f}  max={ws.get('max',0):.4f}")
    print(f"    >= 0.52: {ws.get('above_52',0)*100:.1f}%")

else:
    print("\n유효 조합 없음")
    all_valid_trades = sorted(
        [r for r in results if r.get("n_trades", 0) > 0],
        key=lambda x: -x.get("sharpe", -999)
    )
    print("\n전체 결과 상위 5개 (필터 없음):")
    for r in all_valid_trades[:5]:
        print(f"  TP={r['tp']*100:.0f}% SL={r['sl']*100:.0f}% hold={r['hold']}d  "
              f"AUC={r['oof_auc']:.4f}  수익률={r['total_ret']:+.2f}%  "
              f"샤프={r.get('sharpe',0):.3f}  거래={r['n_trades']}건  "
              f"승률={r.get('win_rate',0):.1f}%  손익비={r.get('pf',0):.2f}")


# ── reversion 비교 테이블 ─────────────────────────────────────────────────
best_r = valid[0] if valid else None
print(f"\n{'='*72}")
print("에이전트 비교 (2026-01-01 ~ 2026-06-19)")
print(f"{'='*72}")
print(f"{'에이전트':>18} {'수익률':>8} {'샤프':>7} {'MDD':>8} {'거래':>6} {'승률':>7} {'손익비':>7}")
print("-" * 72)
rv = REVERSION_BASELINE
print(f"{'reversion':>18}  {rv['total_ret']:>7.2f}%  {rv['sharpe']:>6.3f}  "
      f"{rv['mdd']:>7.2f}%  {rv['n_trades']:>5}건  {rv['win_rate']:>6.1f}%  {rv['pf']:>6.2f}")
if best_r:
    print(f"{'midterm_momentum':>18}  {best_r['total_ret']:>7.2f}%  {best_r['sharpe']:>6.3f}  "
          f"{best_r['mdd']:>7.2f}%  {best_r['n_trades']:>5}건  {best_r['win_rate']:>6.1f}%  "
          f"{best_r['pf']:>6.2f}  (TP={best_r['tp']*100:.0f}%/SL={best_r['sl']*100:.0f}%/hold={best_r['hold']}d)")
else:
    print(f"{'midterm_momentum':>18}  유효 조합 없음")
