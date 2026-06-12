"""
paper_trader.py — 페이퍼 트레이딩 엔진

백테스트와 동일한 체결 가정 하에 신호를 기록하고,
Circuit Breaker / P4 게이트 평가 / 일일 리포트를 제공한다.

LIVE_TRADING=False 상태에서만 동작. 실거래 API 호출 없음.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytz

logger = logging.getLogger(__name__)

KST = pytz.timezone("Asia/Seoul")

# ─── 경로 ────────────────────────────────────────────────────────────────────
_BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
TRADES_PATH     = os.path.join(_BASE_DIR, "paper_trades.json")
POS_PATH        = os.path.join(_BASE_DIR, "paper_positions.json")
META_PATH       = os.path.join(_BASE_DIR, "paper_meta.json")
SNAPSHOT_PATH   = os.path.join(_BASE_DIR, "paper_params_snapshot.json")

# ─── 백테스트 공유 비용 함수 (V7: 별도 구현 금지) ─────────────────────────────
# _barrier_exit()는 OHLCV window 방식 → paper는 EOD Close 방식으로 다름 (명시적 설계 차이)
# _apply_costs()는 수치가 동일하므로 공유해서 drift 방지
from backtest_walkforward import _apply_costs as _bt_apply_costs

# ─── 백테스트와 동일한 파라미터 ────────────────────────────────────────────────
TP_PCT         = 0.15    # +15% ← backtest_walkforward.TP_PCT 와 동기화 필수
SL_PCT         = 0.06    # -6%  ← backtest_walkforward.SL_PCT
MAX_HOLD_DAYS  = 7              # ← backtest_walkforward.HORIZON
ASSUMED_SLIP   = 0.0005  # 가정 슬리피지 0.05% (익일 시초가 지정가 기준)

# ─── 백테스트 기준 EV (V4: config에서 로드, 하드코딩 금지) ─────────────────────
from config import PAPER_BACKTEST_EV_KR as PAPER_BACKTEST_EV
from config import PAPER_BACKTEST_EV_US
BACKTEST_EV    = PAPER_BACKTEST_EV    # KR: +1.468% (TP=15%/SL=6%, G1 채택)
BACKTEST_EV_US = PAPER_BACKTEST_EV_US  # US: None (탐색적 운용 — 백테스트 미검증)

# ─── P3 Circuit Breaker 임계값 ────────────────────────────────────────────────
CB_EV_30     = -0.005    # n≥30: EV ≤ -0.5%
CB_CI_50     = -0.010    # n≥50: CI 하단 < -1.0%
CB_CONSEC    = 8         # 최대 연속 손실 ≥ 8건
CB_AUC       = 0.45      # 분기 평균 AUC < 0.45
CB_GAP       = -0.010    # 페이퍼 EV - 백테스트 EV ≤ -1.0%p
CB_SLIP      = 0.005     # 실측 슬리피지 평균 > 0.50%

# ─── P4 게이트 임계값 ────────────────────────────────────────────────────────
GATE_DAYS    = 60        # 페이퍼 운영 거래일
GATE_N       = 50        # 누적 청산 거래
GATE_EV      = 0.003     # 세후 EV ≥ +0.30%
GATE_WR      = 0.52      # 승률 ≥ 52%
GATE_SLIP    = 0.004     # 실측 슬리피지 평균 < 0.40%
GATE_CONC    = 0.30      # 종목 집중도 < 30%
GATE_CONSEC  = 5         # 최대 연속 손실 ≤ 5건
GATE_AUC     = 0.55      # 분기 평균 AUC ≥ 0.55


# ─────────────────────────────────────────────────────────────────────────────
# JSON 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _is_kr(ticker: str) -> bool:
    return ticker.endswith(".KS") or ticker.endswith(".KQ")


def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def _today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# P1-1. 신호 기록 — log_paper_signal()
# ─────────────────────────────────────────────────────────────────────────────

def log_paper_signal(
    ticker: str,
    name: str,
    agent: str,                    # "momentum" | "reversion" | "eod"
    trigger_types: list[str],
    win_prob: float,
    avg_win: float,
    avg_loss: float,
    rr: float,
    regime_prob: float | None,
    regime_pass: bool,
    entry_price: float | None,     # None → 익일 시초가로 업데이트 예정 (2단계 체결)
    actual_price: float | None,    # 실제 체결가 (호가 스프레드 계산용)
    position_size_pct: float,
    kelly_fraction: float | None,
    auc_at_signal: float | None,   # 신호 시점 레짐 모델 AUC
    eod_close: float | None = None,  # 신호 발생 시 EOD 종가 (갭 슬리피지 계산 기준)
) -> str:
    """
    신호 기록. signal_id 반환.

    entry_price=None 전달 시 2단계 체결 구조:
      포지션 entry_price=None 저장 → update_entry_prices() 호출 후 실제 Open으로 확정.
      확정 전까지 evaluate_positions()에서 청산 체크 스킵.

    entry_price=float 전달 시 기존 동작 (즉시 확정, 테스트 호환).
    """
    signal_id  = str(uuid.uuid4())[:8]
    now        = _now_kst()
    _eod_close = eod_close if eod_close else entry_price  # fallback

    # 실측 슬리피지 (actual 가격이 있을 때)
    actual_slip = None
    if actual_price and entry_price and entry_price > 0:
        actual_slip = abs(actual_price - entry_price) / entry_price

    record = {
        "signal_id":                  signal_id,
        "timestamp":                  now,
        "ticker":                     ticker,
        "name":                       name,
        "agent":                      agent,
        "trigger_types":              trigger_types,
        "win_prob":                   round(win_prob, 4),
        "avg_win":                    round(avg_win, 4),
        "avg_loss":                   round(avg_loss, 4),
        "rr":                         round(rr, 4),
        "regime_prob":                round(regime_prob, 4) if regime_prob is not None else None,
        "regime_pass":                regime_pass,
        "eod_close":                  round(_eod_close, 4) if _eod_close else None,
        "hypothetical_entry_price":   round(entry_price, 4) if entry_price else None,
        "hypothetical_entry_slippage": ASSUMED_SLIP if entry_price else None,
        "actual_price":               round(actual_price, 4) if actual_price else None,
        "actual_slippage":            round(actual_slip, 6) if actual_slip is not None else None,
        "position_size_pct":          round(position_size_pct, 4),
        "kelly_fraction":             round(kelly_fraction, 4) if kelly_fraction is not None else None,
        "auc_at_signal":              round(auc_at_signal, 4) if auc_at_signal is not None else None,
        # 청산 후 채워지는 필드
        "exit_timestamp":             None,
        "exit_price":                 None,
        "exit_reason":                None,   # "TP" | "SL" | "time"
        "raw_pnl_pct":                None,
        "net_pnl_pct":                None,
        "is_win":                     None,
        "holding_days":               None,
        "status":                     "open",
    }

    trades = _load(TRADES_PATH, [])
    trades.append(record)
    _save(TRADES_PATH, trades)

    # 포지션 추가
    positions = _load(POS_PATH, {})
    positions[signal_id] = {
        "signal_id":   signal_id,
        "ticker":      ticker,
        "name":        name,
        "entry_price": entry_price,   # None 허용 — update_entry_prices()로 확정
        "eod_close":   _eod_close,
        "entry_date":  now,
        "trade_days":  0,
        "highest":     entry_price or 0.0,
    }
    _save(POS_PATH, positions)

    # 시작일 기록 + 파라미터 스냅샷 (최초 신호 시 자동)
    meta = _load(META_PATH, {})
    if "start_date" not in meta:
        meta["start_date"] = _today_kst()
        _save(META_PATH, meta)
        snapshot_params()   # V5: 최초 신호 시 파라미터 동결

    logger.info("[Paper] 신호 기록: %s %s %s (레짐=%s)", signal_id, ticker, agent, regime_pass)
    return signal_id


# ─────────────────────────────────────────────────────────────────────────────
# P1-2. 청산 평가 — evaluate_positions()
# 매 거래일 장 종료 후 호출. Triple-Barrier와 동일한 청산 로직.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_positions(price_map: dict[str, float], trade_day: bool = True) -> list[dict]:
    """
    price_map: {ticker: 현재가}
    trade_day: 거래일 1일 경과로 카운트할지 여부 (기본 True)
    반환: 청산된 거래 목록
    """
    positions = _load(POS_PATH, {})
    trades    = _load(TRADES_PATH, [])
    trade_idx = {t["signal_id"]: i for i, t in enumerate(trades)}

    closed = []
    for sig_id, pos in list(positions.items()):
        ticker = pos["ticker"]
        cur    = price_map.get(ticker)
        if cur is None:
            continue

        if trade_day:
            pos["trade_days"] += 1
        pos["highest"] = max(pos["highest"], cur)

        entry = pos.get("entry_price")
        if entry is None:
            continue  # 시초가 미확정 — update_entry_prices() 대기
        raw   = (cur - entry) / entry

        reason = None
        if raw >= TP_PCT:
            reason = "TP"
            exit_price = entry * (1 + TP_PCT)
        elif raw <= -SL_PCT:
            reason = "SL"
            exit_price = entry * (1 - SL_PCT)
        elif pos["trade_days"] >= MAX_HOLD_DAYS:
            reason = "time"
            exit_price = cur

        if reason:
            raw_pnl = (exit_price - entry) / entry
            # V7: backtest_walkforward._apply_costs() 공유 사용 — 별도 재구현 금지
            is_korean = ticker.endswith(".KS") or ticker.endswith(".KQ")
            net_pnl   = _bt_apply_costs(raw_pnl, is_korean)

            idx = trade_idx.get(sig_id)
            if idx is not None:
                trades[idx].update({
                    "exit_timestamp": _now_kst(),
                    "exit_price":     round(exit_price, 4),
                    "exit_reason":    reason,
                    "raw_pnl_pct":    round(raw_pnl * 100, 3),
                    "net_pnl_pct":    round(net_pnl * 100, 3),
                    "is_win":         int(net_pnl > 0),
                    "holding_days":   pos["trade_days"],
                    "status":         "closed",
                })

            closed.append({
                "signal_id": sig_id,
                "ticker":    ticker,
                "name":      pos["name"],
                "reason":    reason,
                "net_pnl":   round(net_pnl * 100, 3),
                "days":      pos["trade_days"],
            })
            del positions[sig_id]
            logger.info("[Paper] 청산: %s %s %s net=%.3f%%", sig_id, ticker, reason, net_pnl * 100)

    _save(POS_PATH, positions)
    _save(TRADES_PATH, trades)
    return closed


# ─────────────────────────────────────────────────────────────────────────────
# 현재가 자동 조회 청산 — evaluate_positions_auto()
# check_ml_positions()와 동일한 주기로 runner에서 호출
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_positions_auto() -> list[dict]:
    """
    paper_positions.json의 open 포지션을 실시간 현재가로 자동 평가·청산.
    장중: yfinance fast_info.last_price (실시간), 장외: 최근 종가 fallback.
    runner.py에서 5분마다 호출 — 실제 매매봇(check_ml_positions)과 동일 주기.
    """
    positions = _load(POS_PATH, {})
    if not positions:
        return []

    tickers   = list({pos["ticker"] for pos in positions.values()})
    price_map: dict[str, float] = {}

    import yfinance as yf
    for ticker in tickers:
        try:
            info  = yf.Ticker(ticker).fast_info
            price = float(info.last_price or 0)
            if price > 0:
                price_map[ticker] = price
        except Exception:
            pass

    if not price_map:
        logger.warning("[Paper] 현재가 조회 실패 — 평가 스킵")

    closed = evaluate_positions(price_map)
    if closed:
        try:
            from notifier import send_telegram
            for c in closed:
                icon = "✅" if c["net_pnl"] > 0 else "❌"
                send_telegram(
                    f"📋 <b>[페이퍼 청산]</b> {c['name']}({c['ticker']})\n"
                    f"사유: {c['reason']} | 세후 {c['net_pnl']:+.3f}% | {c['days']}일 보유"
                )
        except Exception:
            pass

    return closed


# ─────────────────────────────────────────────────────────────────────────────
# AUC 로깅 — log_auc()
# ─────────────────────────────────────────────────────────────────────────────

def log_auc(fold_id: int | str, auc: float):
    """레짐 모델 fold AUC 기록 (H4 모니터링용)."""
    meta = _load(META_PATH, {})
    auc_log = meta.get("auc_log", [])
    auc_log.append({"date": _today_kst(), "fold": fold_id, "auc": round(auc, 4)})
    meta["auc_log"] = auc_log
    _save(META_PATH, meta)

    if auc < 0.50:
        logger.warning("[Paper] AUC 역방향 경고: fold=%s AUC=%.3f", fold_id, auc)
        try:
            from notifier import send_telegram
            send_telegram(
                f"⚠️ <b>[페이퍼] 레짐 AUC 역방향</b>\n"
                f"fold={fold_id}  AUC={auc:.3f}\n"
                f"레짐 예측이 랜덤보다 나쁨 — Circuit Breaker 조건 모니터링"
            )
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# P2. 측정 지표 산출 — get_metrics()
# ─────────────────────────────────────────────────────────────────────────────

def get_metrics(market: str | None = None) -> dict:
    """누적 측정 지표 반환. market='KR'|'US'|None(전체)."""
    trades  = _load(TRADES_PATH, [])
    closed  = [t for t in trades if t["status"] == "closed"]
    if market == "KR":
        closed = [t for t in closed if _is_kr(t["ticker"])]
    elif market == "US":
        closed = [t for t in closed if not _is_kr(t["ticker"])]
    meta    = _load(META_PATH, {})

    if not closed:
        start = meta.get("start_date")
        elapsed = (datetime.now() - datetime.strptime(start, "%Y-%m-%d")).days if start else 0
        return {"n": 0, "ev": 0, "win_rate": 0, "ci_low": 0, "ci_high": 0,
                "max_consec_loss": 0, "max_concentration": 0,
                "avg_actual_slip": None, "backtest_gap": None,
                "elapsed_days": elapsed, "start_date": start}

    pnl_arr = np.array([float(t["net_pnl_pct"]) / 100 for t in closed])
    n       = len(pnl_arr)
    ev      = float(pnl_arr.mean())
    wr      = float(np.mean([t["is_win"] for t in closed]))

    # 부트스트랩 CI
    rng  = np.random.default_rng(42)
    boot = np.array([rng.choice(pnl_arr, size=n, replace=True).mean() for _ in range(2000)])
    ci_low, ci_high = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # 최대 연속 손실
    wins = [t["is_win"] for t in closed]
    max_consec = cur_consec = 0
    for w in wins:
        cur_consec = cur_consec + 1 if w == 0 else 0
        max_consec = max(max_consec, cur_consec)

    # 종목 집중도
    from collections import Counter
    cnt  = Counter(t["ticker"] for t in closed)
    conc = max(cnt.values()) / n if n > 0 else 0

    # 실측 슬리피지 평균
    slips = [float(t["actual_slippage"]) for t in trades
             if t.get("actual_slippage") is not None]
    avg_slip = float(np.mean(slips)) if slips else None

    # 백테스트 갭 (US는 BACKTEST_EV_US=None이므로 갭 계산 불가)
    _ref_ev = BACKTEST_EV_US if market == "US" else BACKTEST_EV
    bt_gap = (ev - _ref_ev) if _ref_ev is not None else None

    # 운영 거래일
    start = meta.get("start_date")
    if start:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        elapsed_days = (datetime.now() - start_dt).days
    else:
        elapsed_days = 0

    return {
        "n":                  n,
        "ev":                 ev,
        "win_rate":           wr,
        "ci_low":             ci_low,
        "ci_high":            ci_high,
        "max_consec_loss":    max_consec,
        "max_concentration":  conc,
        "avg_actual_slip":    avg_slip,
        "backtest_gap":       bt_gap,
        "elapsed_days":       elapsed_days,
        "start_date":         start,
    }


# ─────────────────────────────────────────────────────────────────────────────
# P3. Circuit Breaker — check_circuit_breaker()
# ─────────────────────────────────────────────────────────────────────────────

def check_circuit_breaker(market: str | None = None) -> tuple[bool, str]:
    """
    Circuit Breaker 조건 확인. market='KR'|'US'|None(전체).
    반환: (triggered: bool, reason: str)
    """
    m = get_metrics(market=market)
    n = m["n"]

    checks = []

    if n >= 30 and m["ev"] <= CB_EV_30:
        checks.append(f"EV {m['ev']*100:+.3f}% ≤ -0.5% (n={n})")

    if n >= 50 and m["ci_low"] <= CB_CI_50:
        checks.append(f"CI 하단 {m['ci_low']*100:+.3f}% < -1.0% (n={n})")

    if m["max_consec_loss"] >= CB_CONSEC:
        checks.append(f"최대 연속 손실 {m['max_consec_loss']}건 ≥ {CB_CONSEC}건")

    if n >= 30 and m["backtest_gap"] is not None and m["backtest_gap"] <= CB_GAP:
        checks.append(f"백테스트 갭 {m['backtest_gap']*100:+.3f}%p ≤ -1.0%p (n={n})")

    if m["avg_actual_slip"] is not None and m["avg_actual_slip"] > CB_SLIP:
        checks.append(f"실측 슬리피지 {m['avg_actual_slip']*100:.2f}% > 0.50%")

    # AUC 분기 평균
    meta = _load(META_PATH, {})
    auc_log = meta.get("auc_log", [])
    if auc_log:
        recent_aucs = [e["auc"] for e in auc_log[-12:]]  # 최근 12개 fold
        if recent_aucs:
            avg_auc = float(np.mean(recent_aucs))
            if avg_auc < CB_AUC:
                checks.append(f"분기 평균 AUC {avg_auc:.3f} < 0.45")

    if checks:
        reason = " | ".join(checks)
        logger.warning("[Paper] Circuit Breaker 발동(%s): %s", market or "ALL", reason)
        _alert_circuit_breaker(reason, market=market)
        return True, reason

    return False, ""


def _alert_circuit_breaker(reason: str, market: str | None = None):
    label = f"[페이퍼/{market}]" if market else "[페이퍼]"
    try:
        from notifier import send_telegram
        send_telegram(
            f"🚨 <b>{label} Circuit Breaker 발동</b>\n\n"
            f"사유: {reason}\n\n"
            f"페이퍼 신호 발송을 중단합니다.\n"
            f"원인 분석 후 재시작 여부를 결정하세요."
        )
    except Exception:
        pass


def is_circuit_breaker_active(market: str | None = None) -> bool:
    """Circuit Breaker 상태 확인 (신호 발생 전 게이트). market='KR'|'US'|None."""
    triggered, _ = check_circuit_breaker(market=market)
    return triggered


# ─────────────────────────────────────────────────────────────────────────────
# P4. 실거래 게이트 평가 — evaluate_live_gate()
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_live_gate() -> str:
    """P4 게이트 평가 보고서 반환."""
    m    = get_metrics()
    meta = _load(META_PATH, {})

    lines = ["📊 <b>P4 — 실거래 진입 게이트 평가</b>", ""]

    gates = [
        ("페이퍼 운영 거래일",  m["elapsed_days"],          GATE_DAYS,   "≥", True),
        ("누적 청산 거래",      m["n"],                      GATE_N,      "≥", True),
        ("세후 EV",            m["ev"] * 100,               GATE_EV*100, "≥", True),
        ("CI 하단",            m["ci_low"] * 100,           0,           ">", True),
        ("승률",               m["win_rate"] * 100,         GATE_WR*100, "≥", True),
        ("종목 집중도",         m["max_concentration"]*100,  GATE_CONC*100,"<", False),
        ("최대 연속 손실",      m["max_consec_loss"],         GATE_CONSEC, "≤", False),
    ]

    if m["avg_actual_slip"] is not None:
        gates.append(("실측 슬리피지",  m["avg_actual_slip"]*100, GATE_SLIP*100, "<", False))

    all_pass = True
    for label, val, threshold, op, higher_is_better in gates:
        if op == "≥":
            ok = val >= threshold
        elif op == ">":
            ok = val > threshold
        elif op == "≤":
            ok = val <= threshold
        elif op == "<":
            ok = val < threshold
        else:
            ok = False

        if not ok:
            all_pass = False

        icon = "✅" if ok else "❌"
        fmt_val = "%.1f" % val if isinstance(val, float) else str(int(val))
        fmt_thr = "%.1f" % threshold if isinstance(threshold, float) else str(int(threshold))
        unit = "%" if "%" in label or label in ("EV", "CI 하단", "승률", "슬리피지", "집중도") else ""
        lines.append(f"  {icon} {label:<14}: {fmt_val}{unit} {op} {fmt_thr}{unit} 기준")

    # AUC 조건 (정성)
    auc_log = meta.get("auc_log", [])
    if auc_log:
        recent_aucs = [e["auc"] for e in auc_log[-12:]]
        avg_auc = float(np.mean(recent_aucs))
        auc_ok  = avg_auc >= GATE_AUC
        if not auc_ok:
            all_pass = False
        icon = "✅" if auc_ok else "❌"
        lines.append(f"  {icon} 레짐 AUC 평균   : {avg_auc:.3f} ≥ {GATE_AUC:.2f} 기준")

    lines.append("")
    if all_pass:
        lines.append("✅ <b>모든 게이트 통과 — 실거래 진입 질의 가능</b>")
        lines.append("단, 시작 자금은 전체 운용 자금의 10% 이하부터 권고.")
    else:
        lines.append("❌ <b>게이트 미통과 — LIVE_TRADING=False 유지</b>")
        lines.append("불통과 항목 해소 또는 페이퍼 연장 후 재평가.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# P1-4. 일일 리포트 — daily_report()
# 매 거래일 18:00 자동 호출
# ─────────────────────────────────────────────────────────────────────────────

def _market_metrics_section(trades: list, m: dict, market: str, backtest_ev_val: float | None) -> list[str]:
    """KR 또는 US 누적 지표 섹션 생성."""
    label = "KR 누적 지표" if market == "KR" else "US 누적 지표 (탐색적 — 백테스트 기준 없음)"
    lines: list[str] = [
        "",
        f"<b>{label}</b>",
        f"  거래수: {m['n']}건",
        f"  승률:   {m['win_rate']*100:.1f}%",
        f"  세후EV: {m['ev']*100:+.3f}%",
        f"  CI 95%: [{m['ci_low']*100:+.3f}%, {m['ci_high']*100:+.3f}%]",
        f"  연속손실 최대: {m['max_consec_loss']}건",
        f"  집중도 최대: {m['max_concentration']*100:.0f}%",
    ]
    if m["backtest_gap"] is not None:
        lines.append(f"  백테스트 갭: {m['backtest_gap']*100:+.3f}%p")

    if m["avg_actual_slip"] is not None:
        slip_flag = "✅" if m["avg_actual_slip"] < 0.0015 else ("⚠️" if m["avg_actual_slip"] < 0.003 else "❌")
        lines.append(f"  실측 슬리피지: {m['avg_actual_slip']*100:.3f}% {slip_flag}")

    # 청산 유형별 비율 (2주 페이퍼 핵심 지표)
    closed_mkt = [t for t in trades if t.get("status") == "closed" and
                  (_is_kr(t["ticker"]) if market == "KR" else not _is_kr(t["ticker"]))]
    if closed_mkt:
        n_all = len(closed_mkt)
        lines.append("")
        lines.append(f"<b>{market} 청산 유형</b>")
        for reason, rlabel in [("TP", "TP"), ("SL", "SL"), ("time", "기간만료")]:
            subset = [t for t in closed_mkt if t.get("exit_reason") == reason]
            cnt = len(subset)
            pct = cnt / n_all * 100
            if reason == "time" and subset:
                avg_net = sum(t["net_pnl_pct"] for t in subset) / len(subset)
                flag = "✅" if avg_net >= 1.0 else ("⚠️" if avg_net >= 0.0 else "❌")
                lines.append(f"  {rlabel}: {cnt}건 ({pct:.0f}%)  평균net={avg_net:+.3f}% {flag}")
            else:
                lines.append(f"  {rlabel}: {cnt}건 ({pct:.0f}%)")

    # 진행률 (KR만)
    if market == "KR":
        elapsed  = m.get("elapsed_days") or 0
        lines.append("")
        lines.append(f"  경과 {elapsed}일 / 14일  |  거래 {m['n']}건 / 50건 목표")

    # 백테스트 vs 페이퍼 비교 (KR만, EV가 있을 때)
    if market == "KR" and backtest_ev_val is not None:
        lines += [
            "",
            "<b>백테스트 기준 비교</b>",
            f"  백테스트 EV: {backtest_ev_val*100:+.3f}%",
            f"  페이퍼 EV:   {m['ev']*100:+.3f}%",
        ]
        if m["backtest_gap"] is not None:
            gap = m["backtest_gap"]
            gap_flag = "✅" if gap > -0.005 else ("⚠️" if gap > CB_GAP else "❌")
            lines.append(f"  갭: {gap*100:+.3f}%p {gap_flag}")

    return lines


def daily_report(market: str | None = None) -> str:
    """
    일일 페이퍼 트레이딩 리포트 생성 및 텔레그램 전송.
    market='KR' → KR 섹션만, 'US' → US 섹션만, None → KR+US 전체.
    """
    trades    = _load(TRADES_PATH, [])
    positions = _load(POS_PATH, {})
    today     = _today_kst()

    # 오늘 신호 (market 필터 적용)
    all_signals   = [t for t in trades if t["timestamp"].startswith(today)]
    all_closed_td = [t for t in trades
                     if t["status"] == "closed" and
                     t.get("exit_timestamp", "").startswith(today)]

    if market == "KR":
        today_signals = [s for s in all_signals   if _is_kr(s["ticker"])]
        today_closed  = [c for c in all_closed_td if _is_kr(c["ticker"])]
        pos_filtered  = {k: v for k, v in positions.items() if _is_kr(v["ticker"])}
        mkt_label     = "KR"
    elif market == "US":
        today_signals = [s for s in all_signals   if not _is_kr(s["ticker"])]
        today_closed  = [c for c in all_closed_td if not _is_kr(c["ticker"])]
        pos_filtered  = {k: v for k, v in positions.items() if not _is_kr(v["ticker"])}
        mkt_label     = "US"
    else:
        today_signals = all_signals
        today_closed  = all_closed_td
        pos_filtered  = positions
        mkt_label     = "KR+US"

    kr_sig = [s for s in today_signals if _is_kr(s["ticker"])]
    us_sig = [s for s in today_signals if not _is_kr(s["ticker"])]
    sig_detail = (f"KR: {len(kr_sig)}건 / US: {len(us_sig)}건"
                  if market is None else f"{market}: {len(today_signals)}건")

    lines = [
        f"📋 <b>[페이퍼/{mkt_label}] 일일 리포트 {today}</b>",
        "",
        f"당일 신호: {len(today_signals)}건 ({sig_detail})",
    ]

    if today_signals:
        for s in today_signals:
            lines.append(
                f"  • {s['name']}({s['ticker']}) "
                f"레짐={s['regime_pass']} "
                f"트리거={','.join(s['trigger_types'])}"
            )

    lines.append(f"\n당일 청산: {len(today_closed)}건")
    for c in today_closed:
        icon = "✅" if c["is_win"] else "❌"
        lines.append(
            f"  {icon} {c['name']}({c['ticker']}) "
            f"{c['exit_reason']} {c['net_pnl_pct']:+.3f}%"
        )

    # 미청산 포지션 (현재가·수익률 포함)
    lines.append(f"\n미청산 포지션: {len(pos_filtered)}건")
    for pos in pos_filtered.values():
        entry = pos.get("entry_price")
        if entry:
            try:
                import yfinance as yf
                cur = yf.Ticker(pos["ticker"]).fast_info.last_price or 0
                pnl = (cur - entry) / entry * 100
                arrow = "▲" if pnl >= 0 else "▼"
                lines.append(
                    f"  • {pos['name']}({pos['ticker']}) "
                    f"진입 {entry:,.0f}원 → 현재 {cur:,.0f}원 "
                    f"{arrow}{abs(pnl):.2f}% ({pos['trade_days']}일)"
                )
            except Exception:
                lines.append(
                    f"  • {pos['name']}({pos['ticker']}) "
                    f"진입 {entry:,.0f}원 ({pos['trade_days']}일 경과)"
                )
        else:
            lines.append(
                f"  • {pos['name']}({pos['ticker']}) "
                f"시초가 미확정 — 내일 진입 예정"
            )

    # ── KR 섹션 ──────────────────────────────────────────────────────────────
    if market in (None, "KR"):
        m_kr = get_metrics(market="KR")
        lines.extend(_market_metrics_section(trades, m_kr, "KR", BACKTEST_EV))
        cb_kr, cb_kr_reason = check_circuit_breaker(market="KR")
        if cb_kr:
            lines += ["", f"🚨 <b>KR Circuit Breaker 발동</b>: {cb_kr_reason}"]
        else:
            lines.append("\n✅ KR Circuit Breaker: 정상")

    # ── US 섹션 ──────────────────────────────────────────────────────────────
    if market in (None, "US"):
        m_us = get_metrics(market="US")
        if m_us["n"] > 0 or market == "US":
            lines.extend(_market_metrics_section(trades, m_us, "US", BACKTEST_EV_US))
            if market == "US":
                lines.append("  (백테스트 기준 없음 — 탐색적 운용)")
            cb_us, cb_us_reason = check_circuit_breaker(market="US")
            if cb_us:
                lines += ["", f"🚨 <b>US Circuit Breaker 발동</b>: {cb_us_reason}"]
            else:
                lines.append("\n✅ US Circuit Breaker: 정상")

    # V5: 파라미터 drift 체크
    drifts = check_param_drift()
    if drifts:
        lines += ["", "<b>⚠️ 파라미터 변경 감지 (카운트 리셋 필요)</b>"]
        lines.extend(f"  {d}" for d in drifts)

    # V6: 로직 변경 이력
    meta    = _load(META_PATH, {})
    history = meta.get("logic_change_history", [])
    if history:
        last   = history[-1]
        n_days = (datetime.now() - datetime.strptime(meta.get("start_date", today), "%Y-%m-%d")).days
        lines.append(f"\n로직 변경 후 {n_days}일째 (최근 변경: {last['date']} — {last['reason']})")

    report = "\n".join(l for l in lines if l is not None)

    try:
        from notifier import send_telegram
        send_telegram(report)
    except Exception as e:
        logger.warning("[Paper] 일일 리포트 전송 실패: %s", e)

    logger.info("[Paper] 일일 리포트 완료 (%s market=%s)", today, market or "ALL")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# V5 — 파라미터 동결 메커니즘
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_params() -> dict:
    """
    페이퍼 시작 시 파라미터를 paper_params_snapshot.json에 저장.
    log_paper_signal() 최초 호출 시 자동 실행됨.
    """
    from config import ML_MIN_WIN_PROB, ML_MIN_RISK_REWARD
    params = {
        "snapshot_date":    _today_kst(),
        "MIN_WIN_PROB":     ML_MIN_WIN_PROB,
        "MIN_RR":           ML_MIN_RISK_REWARD,
        "regime_threshold": 0.50,
        "TP_PCT":           TP_PCT,
        "SL_PCT":           SL_PCT,
        "HORIZON":          MAX_HOLD_DAYS,
        "BACKTEST_EV":      BACKTEST_EV,
    }
    _save(SNAPSHOT_PATH, params)
    logger.info("[Paper] 파라미터 스냅샷 저장: %s", params)
    return params


def check_param_drift() -> list[str]:
    """
    현재 파라미터와 스냅샷 비교. 불일치 시 경고 목록 반환.
    daily_report()에서 매일 자동 호출.
    """
    snap = _load(SNAPSHOT_PATH, {})
    if not snap:
        return ["⚠️ 스냅샷 없음 — snapshot_params() 먼저 실행"]

    from config import ML_MIN_WIN_PROB, ML_MIN_RISK_REWARD
    current = {
        "MIN_WIN_PROB":     ML_MIN_WIN_PROB,
        "MIN_RR":           ML_MIN_RISK_REWARD,
        "regime_threshold": 0.50,
        "TP_PCT":           TP_PCT,
        "SL_PCT":           SL_PCT,
        "HORIZON":          MAX_HOLD_DAYS,
    }

    drifts = []
    for key, snap_val in snap.items():
        if key in ("snapshot_date", "BACKTEST_EV"):
            continue
        cur_val = current.get(key)
        if cur_val is not None and abs(float(cur_val) - float(snap_val)) > 1e-9:
            drifts.append(
                f"🚨 {key}: {snap_val} → {cur_val} (페이퍼 중 변경! 카운트 리셋 필요)"
            )
    return drifts


# ─────────────────────────────────────────────────────────────────────────────
# V6 — 시작일 + 카운트 리셋 메커니즘
# ─────────────────────────────────────────────────────────────────────────────

def register_logic_change(reason: str):
    """
    신호 로직 변경 시 호출.
    paper_start_date를 오늘로 리셋 + reason 로그 저장.
    daily_report에 "로직 변경 후 N일째" 표시.
    """
    meta     = _load(META_PATH, {})
    old_start = meta.get("start_date", "없음")
    today    = _today_kst()

    history = meta.get("logic_change_history", [])
    history.append({
        "date":             today,
        "reason":           reason,
        "prev_start_date":  old_start,
        "trades_archived":  len([t for t in _load(TRADES_PATH, []) if t.get("status") == "closed"]),
    })
    meta["logic_change_history"] = history
    meta["start_date"]           = today
    _save(META_PATH, meta)

    # 파라미터 스냅샷도 갱신
    snapshot_params()

    logger.warning("[Paper] 로직 변경 등록 — 카운트 리셋 (이전 시작일: %s): %s", old_start, reason)
    try:
        from notifier import send_telegram
        send_telegram(
            f"⚠️ <b>[페이퍼] 로직 변경 등록</b>\n"
            f"사유: {reason}\n"
            f"이전 시작일: {old_start} → 오늘({today})로 리셋\n"
            f"게이트 평가 카운트가 오늘부터 재시작됩니다."
        )
    except Exception:
        pass


def update_entry_prices(market: str) -> list[str]:
    """
    KR: 09:05 KST, US: ET 09:35 이후 호출.
    entry_price=None인 포지션에 실제 시초가(Open)를 기록.
    반환: 업데이트된 종목 요약 목록.
    """
    positions = _load(POS_PATH, {})
    trades    = _load(TRADES_PATH, [])

    pending = {
        k: v for k, v in positions.items()
        if v.get("entry_price") is None
        and (_is_kr(v["ticker"]) if market == "KR" else not _is_kr(v["ticker"]))
    }
    if not pending:
        return []

    tickers   = list({pos["ticker"] for pos in pending.values()})
    open_map: dict[str, float] = {}

    try:
        import FinanceDataReader as fdr
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        for ticker in tickers:
            try:
                df = fdr.DataReader(ticker, today)
                if not df.empty and "Open" in df.columns:
                    val = float(df["Open"].iloc[-1])
                    if val > 0:
                        open_map[ticker] = val
            except Exception:
                pass
    except ImportError:
        try:
            import yfinance as yf
            for ticker in tickers:
                try:
                    data = yf.download(ticker, period="1d", progress=False, auto_adjust=True)
                    if not data.empty:
                        try:
                            import pandas as pd
                            if isinstance(data.columns, pd.MultiIndex):
                                data.columns = data.columns.get_level_values(0)
                        except Exception:
                            pass
                        val = float(data["Open"].iloc[-1])
                        if val > 0:
                            open_map[ticker] = val
                except Exception:
                    pass
        except ImportError:
            logger.warning("[Paper] 시초가 조회 라이브러리 없음 (fdr/yf)")
            return []

    trade_idx = {t["signal_id"]: i for i, t in enumerate(trades)}
    updated: list[str] = []

    for sig_id, pos in pending.items():
        ticker     = pos["ticker"]
        open_price = open_map.get(ticker)
        if not open_price:
            logger.warning("[Paper] 시초가 조회 실패 — %s 미업데이트", ticker)
            continue

        # 포지션 확정
        positions[sig_id]["entry_price"] = round(open_price, 4)
        positions[sig_id]["highest"]     = round(open_price, 4)

        # trades 업데이트 + 실제 갭 슬리피지 계산
        idx = trade_idx.get(sig_id)
        if idx is not None:
            eod_close = trades[idx].get("eod_close")
            gap_slip  = abs(open_price - eod_close) / eod_close if eod_close else None
            trades[idx]["hypothetical_entry_price"]    = round(open_price, 4)
            trades[idx]["hypothetical_entry_slippage"] = round(gap_slip, 6) if gap_slip is not None else None

        updated.append(f"{pos.get('name', ticker)}({ticker}) {open_price:,.0f}")
        logger.info("[Paper] 시초가 확정: %s open=%.4f", ticker, open_price)

    if updated:
        _save(POS_PATH, positions)
        _save(TRADES_PATH, trades)
        try:
            from notifier import send_telegram
            send_telegram(
                f"📌 <b>[페이퍼/{market}] 시초가 진입 확정</b>\n"
                + "\n".join(f"  • {u}" for u in updated)
            )
        except Exception:
            pass

    return updated


def get_start_date() -> str | None:
    """현재 페이퍼 시작일 반환."""
    return _load(META_PATH, {}).get("start_date")


# ─────────────────────────────────────────────────────────────────────────────
# 주차별 요약 — weekly_summary()
# ─────────────────────────────────────────────────────────────────────────────

def weekly_summary() -> str:
    """매주 일요일 집계 리포트."""
    m = get_metrics()

    lines = [
        "📊 <b>[페이퍼] 주차별 집계</b>",
        "",
        f"  거래수: {m['n']}건 (목표 누적 {GATE_N}건)",
        f"  세후EV: {m['ev']*100:+.3f}% (목표 ≥{GATE_EV*100:.2f}%)",
        f"  CI 95%: [{m['ci_low']*100:+.3f}%, {m['ci_high']*100:+.3f}%]",
        f"  승률:   {m['win_rate']*100:.1f}% (목표 ≥{GATE_WR*100:.0f}%)",
        f"  연속손실: {m['max_consec_loss']}건 (기준 ≤{GATE_CONSEC}건)",
        f"  집중도: {m['max_concentration']*100:.0f}% (기준 <{GATE_CONC*100:.0f}%)",
        f"  백테스트 갭: {m['backtest_gap']*100:+.3f}%p" if m['backtest_gap'] is not None else "",
        f"  운영 {m['elapsed_days']}거래일 경과 (목표 {GATE_DAYS}일)",
    ]

    try:
        from notifier import send_telegram
        send_telegram("\n".join(l for l in lines if l))
    except Exception:
        pass

    return "\n".join(l for l in lines if l)
