#!/usr/bin/env python3
"""
trend_agent.py — 규칙 기반 추세 추종 에이전트

ML 없이 순수 규칙: ADX + 이동평균 정배열 + 트레일링 스톱
그리드 서치: ADX [20,25,30] × trail [1.5,2.0,2.5ATR] × vol [1.0,1.3,1.5] = 27개
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import itertools
import numpy as np
import pandas as pd
import yfinance as yf

# ── 파라미터 ──────────────────────────────────────────────────────────────
BT_START      = "2023-01-01"
BT_END        = "2026-06-19"
POSITION_SIZE = 0.10
ATR_RISK_PCT  = 0.01        # ATR 기반 리스크 한도 1%
MAX_POSITIONS = 10
COST_RT       = 0.0046
ADX_EXIT      = 20
MAX_HOLD      = 60          # 최대 보유 거래일

ADX_LIST   = [20, 25, 30]
TRAIL_LIST = [1.5, 2.0, 2.5]
VOL_LIST   = [1.0, 1.3, 1.5]

MIN_SHARPE = 1.0
MAX_MDD    = -20.0
MIN_TRADES = 30
MIN_PF     = 1.5

REVERSION_BT = {   # 2026-01-01 ~ 2026-06-19 기준
    "total_ret": 20.86, "sharpe": 1.976, "mdd": -7.16,
    "n_trades": 84, "win_rate": 59.5, "pf": 1.59,
    "monthly": {
        "2026-02": 0.00, "2026-03": -0.50,
        "2026-04": 0.38, "2026-05": -2.98, "2026-06": 0.00,
    }
}


# ════════════════════════════════════════════════════════════════════════════
# 지표 계산
# ════════════════════════════════════════════════════════════════════════════
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    for p in [5, 20, 60, 200]:
        df[f"ma{p}"] = c.rolling(p).mean()

    # True Range / ATR (Wilder 스무딩)
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["tr"]      = tr
    df["atr"]     = tr.ewm(com=13, adjust=False).mean()
    df["vol_ma20"] = v.rolling(20).mean()

    # ADX (Wilder 스무딩)
    dm_p = (h - h.shift(1)).clip(lower=0)
    dm_m = (l.shift(1) - l).clip(lower=0)
    # 한쪽이 더 클 때만 유효
    dm_p = dm_p.where(dm_p >= dm_m, 0.0)
    dm_m = dm_m.where(dm_m >  dm_p, 0.0)

    atr_w    = tr.ewm(com=13, adjust=False).mean()
    di_p     = 100 * dm_p.ewm(com=13, adjust=False).mean() / atr_w
    di_m     = 100 * dm_m.ewm(com=13, adjust=False).mean() / atr_w
    dx       = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    df["adx"] = dx.ewm(com=13, adjust=False).mean()

    return df


def _strip_tz(idx):
    return idx.tz_localize(None) if idx.tzinfo else idx


def _row(df, date):
    if df is None: return pd.DataFrame()
    idx = _strip_tz(df.index)
    pos = np.where(idx == date)[0]
    return df.iloc[pos] if len(pos) else pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# 백테스트 (단일 파라미터 조합)
# ════════════════════════════════════════════════════════════════════════════
def run_backtest(
    ind_data:   dict[str, pd.DataFrame],
    raw_data:   dict[str, pd.DataFrame],
    kospi_ma200: pd.Series,
    kospi_close: pd.Series,
    all_days:    pd.DatetimeIndex,
    adx_thresh:  float,
    trail_mult:  float,
    vol_mult:    float,
) -> dict:
    bt_days = all_days[(all_days >= BT_START) & (all_days <= BT_END)]
    if len(bt_days) == 0:
        return {"n_trades": 0}

    # ── 신호 사전 생성 ────────────────────────────────────────────────────
    signals_by_date: dict[pd.Timestamp, list[dict]] = {}
    for ticker, df_i in ind_data.items():
        idx = _strip_tz(df_i.index)
        for date in bt_days:
            pos = np.where(idx == date)[0]
            if len(pos) == 0: continue
            r = df_i.iloc[pos[0]]
            if pd.isna(r.get("adx", np.nan)): continue
            if pd.isna(r.get("ma200", np.nan)): continue
            # 진입 조건
            if r["adx"] < adx_thresh: continue
            if not (r["ma5"] > r["ma20"] > r["ma60"] > r["ma200"]): continue
            if pd.isna(r.get("vol_ma20", np.nan)) or r["vol_ma20"] == 0: continue
            if r["Volume"] < r["vol_ma20"] * vol_mult: continue
            # KOSPI MA200 위
            km200 = kospi_ma200.get(date, np.nan)
            kc    = kospi_close.get(date, np.nan)
            if pd.isna(km200) or pd.isna(kc) or kc <= km200: continue
            signals_by_date.setdefault(date, []).append({
                "ticker": ticker,
                "atr":    float(r["atr"]),
            })

    # ── 시뮬레이션 ────────────────────────────────────────────────────────
    capital   = 1.0
    positions: list[dict] = []
    trades:    list[dict] = []
    daily_nav: list[tuple] = []   # (date, nav)
    exit_queue: list[str] = []    # 당일 청산 예정 ticker 목록

    for i, today in enumerate(bt_days):
        next_day = bt_days[i + 1] if i + 1 < len(bt_days) else None

        # ── 전날 대기 중인 청산 처리 (익일 시초가) ───────────────────────
        still_open = []
        for pos in positions:
            if pos.get("_exit_next"):
                df_r = raw_data.get(pos["ticker"])
                tr   = _row(df_r, today)
                if tr.empty:
                    pos.pop("_exit_next", None)
                    still_open.append(pos); continue
                ep    = pos["entry_price"]
                xp    = float(tr["Open"].iloc[0])
                net   = xp / ep - 1 - COST_RT
                capital += pos["size"] * (1 + net)
                trades.append({
                    "net_ret":    net,
                    "exit_reason": pos["_exit_reason"],
                    "win":        net > 0,
                    "hold_days":  pos["hold_days"],
                })
            else:
                still_open.append(pos)
        positions = still_open

        # ── 보유 포지션 업데이트 + 청산 조건 체크 ───────────────────────
        still_open = []
        for pos in positions:
            pos["hold_days"] += 1
            df_r     = raw_data.get(pos["ticker"])
            today_r  = _row(df_r, today)
            df_i     = ind_data.get(pos["ticker"])
            today_i  = _row(df_i, today) if df_i is not None else pd.DataFrame()

            if today_r.empty or today_i.empty:
                still_open.append(pos); continue

            hi  = float(today_r["High"].iloc[0])
            lo  = float(today_r["Low"].iloc[0])
            cl  = float(today_r["Close"].iloc[0])
            ep  = pos["entry_price"]
            pos["high_since_entry"] = max(pos["high_since_entry"], hi)

            trail_stop = pos["high_since_entry"] - trail_mult * pos["atr_at_entry"]
            ma20       = float(today_i["ma20"].iloc[0]) if not pd.isna(today_i["ma20"].iloc[0]) else ep
            adx_now    = float(today_i["adx"].iloc[0])  if not pd.isna(today_i["adx"].iloc[0])  else 99

            # 트레일링 스톱 (EOD 기준, 익일 시초가 청산)
            exit_rsn = None
            if cl <= trail_stop:
                exit_rsn = "TRAIL"
            elif cl < ma20:
                exit_rsn = "MA20"
            elif adx_now < ADX_EXIT:
                exit_rsn = "ADX"
            elif pos["hold_days"] >= MAX_HOLD:
                exit_rsn = "TIME"

            if exit_rsn and next_day is not None:
                pos["_exit_next"]   = True
                pos["_exit_reason"] = exit_rsn
            still_open.append(pos)
        positions = still_open

        # ── 신규 진입 ─────────────────────────────────────────────────────
        if next_day is not None and today in signals_by_date:
            for sig in signals_by_date[today]:
                if len(positions) >= MAX_POSITIONS: break
                if any(p["ticker"] == sig["ticker"] for p in positions): continue
                df_r = raw_data.get(sig["ticker"])
                if df_r is None: continue
                entry_r = _row(df_r, next_day)
                if entry_r.empty: continue
                ep = float(entry_r["Open"].iloc[0])
                if ep <= 0: continue

                base_alloc = capital * POSITION_SIZE
                atr_val    = sig["atr"]
                if atr_val > 0:
                    atr_alloc = (capital * ATR_RISK_PCT * ep) / (2 * atr_val)
                    size = min(base_alloc, atr_alloc)
                else:
                    size = base_alloc
                size = max(size, 0)

                capital -= size * (1 + COST_RT / 2)
                positions.append({
                    "ticker":           sig["ticker"],
                    "entry_price":      ep,
                    "hold_days":        0,
                    "size":             size,
                    "atr_at_entry":     atr_val,
                    "high_since_entry": ep,
                })

        # ── 일별 NAV ─────────────────────────────────────────────────────
        pos_mkt = 0.0
        for pos in positions:
            df_r = raw_data.get(pos["ticker"])
            tr   = _row(df_r, today)
            cp   = float(tr["Close"].iloc[0]) if not tr.empty else pos["entry_price"]
            pos_mkt += pos["size"] * (cp / pos["entry_price"])
        daily_nav.append((today, capital + pos_mkt))

    if not trades or len(daily_nav) < 2:
        return {"n_trades": 0}

    df_tr  = pd.DataFrame(trades)
    nav_s  = pd.Series(
        [v for _, v in daily_nav],
        index=pd.DatetimeIndex([d for d, _ in daily_nav])
    )
    dr     = nav_s.pct_change().dropna()
    wins   = df_tr[df_tr["net_ret"] > 0]["net_ret"]
    losses = df_tr[df_tr["net_ret"] <= 0]["net_ret"]

    # 월별 수익률
    monthly = {}
    mon_groups = nav_s.resample("ME")
    prev_val   = nav_s.iloc[0]
    for period, group in mon_groups:
        if group.empty: continue
        cur_val = group.iloc[-1]
        mon_key = period.strftime("%Y-%m")
        monthly[mon_key] = round((cur_val / prev_val - 1) * 100, 2)
        prev_val = cur_val

    return {
        "n_trades":  len(df_tr),
        "total_ret": round((nav_s.iloc[-1] - 1) * 100, 2),
        "sharpe":    round(dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0, 3),
        "mdd":       round(((nav_s - nav_s.cummax()) / nav_s.cummax()).min() * 100, 2),
        "win_rate":  round(df_tr["win"].mean() * 100, 1),
        "avg_win":   round(wins.mean()   * 100, 2) if len(wins)   > 0 else 0.0,
        "avg_loss":  round(losses.mean() * 100, 2) if len(losses) > 0 else 0.0,
        "avg_hold":  round(df_tr["hold_days"].mean(), 1),
        "pf":        round(abs(wins.mean() / losses.mean()), 2)
                     if len(losses) > 0 and losses.mean() != 0 else 0.0,
        "exit_counts": df_tr["exit_reason"].value_counts().to_dict(),
        "monthly":   monthly,
        "nav":       nav_s,
    }


if __name__ == "__main__":
    # 그리드 서치 백테스트는 이 파일을 직접 실행할 때만 동작한다.
    # (compute_indicators 등을 import할 때 200종목 다운로드·백테스트가 도는 부작용 방지)
    # ════════════════════════════════════════════════════════════════════════════
    # 데이터 준비
    # ════════════════════════════════════════════════════════════════════════════
    print("KOSPI 다운로드 (4년치)...")
    kospi_raw = yf.download("^KS11", period="4y", auto_adjust=True, progress=False)
    if isinstance(kospi_raw.columns, pd.MultiIndex):
        kospi_raw.columns = kospi_raw.columns.get_level_values(0)
    kospi_raw.index = _strip_tz(kospi_raw.index)
    kospi_ind    = compute_indicators(kospi_raw)
    kospi_ma200  = kospi_ind["ma200"].dropna()
    kospi_close  = kospi_raw["Close"]

    print("유니버스 로드...")
    from signals.krx_universe import get_krx_backtest_universe
    tickers = list(get_krx_backtest_universe(top_n=200).keys())

    print(f"티커 데이터 다운로드 ({len(tickers)}개)...")
    raw_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="4y", auto_adjust=True, progress=False)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = _strip_tz(df.index)
            df = df[["Open","High","Low","Close","Volume"]].dropna()
            raw_data[ticker] = df
        except Exception:
            pass
    print(f"다운로드 완료: {len(raw_data)}개")

    print("지표 계산...")
    ind_data: dict[str, pd.DataFrame] = {}
    for ticker, df in raw_data.items():
        try:
            df_i = compute_indicators(df)
            # Volume을 ind_data에도 포함 (진입 조건 체크용)
            df_i["Volume"] = df["Volume"]
            if len(df_i.dropna(subset=["ma200","adx"])) >= 20:
                ind_data[ticker] = df_i
        except Exception:
            pass
    print(f"지표 계산 완료: {len(ind_data)}개")

    all_days = _strip_tz(kospi_raw.index)


    # ════════════════════════════════════════════════════════════════════════════
    # 그리드 서치
    # ════════════════════════════════════════════════════════════════════════════
    combos = list(itertools.product(ADX_LIST, TRAIL_LIST, VOL_LIST))
    print(f"\n그리드 서치: {len(combos)}개 조합...")

    results = []
    for i, (adx, trail, vol) in enumerate(combos, 1):
        print(f"  [{i:2d}/{len(combos)}] ADX>={adx}  trail={trail}ATR  vol>{vol}x ...",
              end=" ", flush=True)
        r = run_backtest(ind_data, raw_data, kospi_ma200, kospi_close,
                         all_days, adx, trail, vol)
        r.update({"adx": adx, "trail": trail, "vol": vol})
        results.append(r)
        if r.get("n_trades", 0) > 0:
            print(f"수익률={r['total_ret']:+.2f}%  샤프={r['sharpe']:.3f}  "
                  f"MDD={r['mdd']:.2f}%  거래={r['n_trades']}건")
        else:
            print("거래없음")


    # ════════════════════════════════════════════════════════════════════════════
    # 결과 출력
    # ════════════════════════════════════════════════════════════════════════════
    valid = [
        r for r in results
        if r.get("n_trades", 0) >= MIN_TRADES
        and r.get("sharpe", -999) >= MIN_SHARPE
        and r.get("mdd", -999) >= MAX_MDD
        and r.get("pf", 0) >= MIN_PF
    ]
    valid.sort(key=lambda x: -x.get("sharpe", -999))

    print(f"\n{'='*88}")
    print(f"추세 추종 에이전트 그리드 서치  ({BT_START} ~ {BT_END})")
    print(f"유효 조합: {len(valid)} / {len(results)}개"
          f"  (샤프>={MIN_SHARPE}  MDD>={MAX_MDD}%  거래>={MIN_TRADES}건  손익비>={MIN_PF})")
    print(f"{'='*88}")

    if valid:
        print(f"\n{'ADX':>4} {'trail':>6} {'vol':>5} {'수익률':>8} {'샤프':>7} {'MDD':>8} "
              f"{'거래':>5} {'승률':>7} {'손익비':>7} {'평균보유':>8}")
        print("-" * 88)
        for r in valid:
            print(f"{r['adx']:>4}  {r['trail']:>5.1f}x  {r['vol']:>4.1f}x"
                  f"  {r['total_ret']:>7.2f}%  {r['sharpe']:>6.3f}  {r['mdd']:>7.2f}%"
                  f"  {r['n_trades']:>5}  {r['win_rate']:>6.1f}%  {r['pf']:>6.2f}"
                  f"  {r['avg_hold']:>6.1f}d")

        # ── 상위 5개 상세 ─────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("상위 5개 조합 상세")
        print(f"{'='*60}")
        for rank, r in enumerate(valid[:5], 1):
            print(f"\n[{rank}위] ADX>={r['adx']}  trail={r['trail']}ATR  vol>{r['vol']}x")
            print(f"  총 수익률   : {r['total_ret']:+.2f}%")
            print(f"  샤프 비율   : {r['sharpe']:.3f}")
            print(f"  MDD         : {r['mdd']:.2f}%")
            print(f"  거래 수     : {r['n_trades']}건")
            print(f"  승률        : {r['win_rate']:.1f}%")
            print(f"  평균 수익   : {r['avg_win']:+.2f}%")
            print(f"  평균 손실   : {r['avg_loss']:+.2f}%")
            print(f"  손익비      : {r['pf']:.2f}")
            print(f"  평균 보유   : {r['avg_hold']:.1f}일")
            print(f"  청산 사유   : {r['exit_counts']}")

        # ── 최적 조합 월별 수익률 ─────────────────────────────────────────────
        best = valid[0]
        monthly = best.get("monthly", {})
        print(f"\n{'='*60}")
        print(f"최적 조합 월별 수익률  (ADX>={best['adx']}  trail={best['trail']}ATR  vol>{best['vol']}x)")
        print(f"{'='*60}")
        years = sorted({k[:4] for k in monthly})
        for year in years:
            yr_months = {k: v for k, v in monthly.items() if k.startswith(year)}
            cumulative = sum(yr_months.values())
            month_str  = "  ".join(f"{k[5:]}월:{v:+.1f}%" for k, v in sorted(yr_months.items()))
            print(f"  {year}: {month_str}  ← 합계 {cumulative:+.2f}%")

        # ── reversion 비교 ───────────────────────────────────────────────────
        print(f"\n{'='*72}")
        print("에이전트 비교 (2023-01-01 ~ 2026-06-19 / reversion은 2026 기준)")
        print(f"{'='*72}")
        rv = REVERSION_BT
        fmt = "  {:<22s}  {:>9}  {:>7}  {:>8}  {:>6}  {:>7}  {:>7}"
        print(fmt.format("에이전트", "수익률", "샤프", "MDD", "거래", "승률", "손익비"))
        print("  " + "-" * 68)
        print(fmt.format(
            "reversion (2026만)",
            f"{rv['total_ret']:+.2f}%", f"{rv['sharpe']:.3f}",
            f"{rv['mdd']:.2f}%", f"{rv['n_trades']}건",
            f"{rv['win_rate']:.1f}%", f"{rv['pf']:.2f}",
        ))
        print(fmt.format(
            f"trend (ADX{best['adx']}/T{best['trail']}/V{best['vol']})",
            f"{best['total_ret']:+.2f}%", f"{best['sharpe']:.3f}",
            f"{best['mdd']:.2f}%", f"{best['n_trades']}건",
            f"{best['win_rate']:.1f}%", f"{best['pf']:.2f}",
        ))

        # ── 월별 수익률 상관계수 (2026 공통 구간) ────────────────────────────
        rv_monthly = rv["monthly"]
        trend_monthly = {k: v for k, v in monthly.items() if k in rv_monthly}
        if len(trend_monthly) >= 3:
            keys = sorted(trend_monthly)
            t_vals = [trend_monthly[k] for k in keys]
            r_vals = [rv_monthly[k] for k in keys]
            corr   = float(np.corrcoef(t_vals, r_vals)[0, 1])
            print(f"\n  월별 수익률 상관계수 (2026 공통 {len(keys)}개월): {corr:.3f}")
            if   corr <  0:   interp = "역상관 — 분산 효과 우수"
            elif corr < 0.3:  interp = "약한 양의 상관 — 분산 효과 양호"
            elif corr < 0.6:  interp = "중간 상관 — 분산 효과 보통"
            else:             interp = "강한 양의 상관 — 분산 효과 제한적"
            print(f"  → {interp}")
            print(f"\n  {'월':>8}  {'trend':>8}  {'reversion':>10}")
            for k in keys:
                print(f"  {k:>8}  {trend_monthly[k]:>+7.2f}%  {rv_monthly[k]:>+9.2f}%")
        else:
            print("\n  (공통 구간 부족 — 상관계수 계산 불가)")

    else:
        print("\n유효 조합 없음. 전체 결과 상위 5개 (필터 없음):")
        all_valid = sorted(
            [r for r in results if r.get("n_trades", 0) > 0],
            key=lambda x: -x.get("sharpe", -999)
        )
        for r in all_valid[:5]:
            print(f"  ADX>={r['adx']}  trail={r['trail']}ATR  vol>{r['vol']}x  "
                  f"수익률={r['total_ret']:+.2f}%  샤프={r['sharpe']:.3f}  "
                  f"MDD={r['mdd']:.2f}%  거래={r['n_trades']}건")
