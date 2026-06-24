"""
paper_trader.py — 페이퍼 트레이딩 엔진

백테스트와 동일한 체결 가정 하에 신호를 기록하고,
Circuit Breaker / P4 게이트 평가 / 일일 리포트를 제공한다.

슬롯 분리 10+10 전략 (2026-06-19 채택):
  reversion 전용 10슬롯 + trend 전용 10슬롯, 서로 침범 불가.

LIVE_TRADING=False 상태에서만 동작. 실거래 API 호출 없음.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
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

# ─── 백테스트 공유 비용 함수 (별도 구현 금지) ────────────────────────────────
from backtest_walkforward import _apply_costs as _bt_apply_costs

# ─── config에서 파라미터 로드 ────────────────────────────────────────────────
from config import (
    PAPER_BACKTEST_EV_KR as PAPER_BACKTEST_EV,
    TP_PCT, SL_PCT, EOD_SLIPPAGE_PCT, EOD_HORIZON,
    REV_SLOTS, TR_SLOTS,
)
PAPER_BACKTEST_EV_US = None  # US 미운용

BACKTEST_EV    = PAPER_BACKTEST_EV    # KR 백테스트 참고 EV
MAX_HOLD_DAYS  = EOD_HORIZON          # reversion 보유기간 10거래일
ASSUMED_SLIP   = EOD_SLIPPAGE_PCT     # 가정 슬리피지 0.05%
# trend 에이전트: TP 없음, trailing stop 2.0×ATR + MA20 이탈 청산 (evaluate_positions에서 별도 처리)

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
# 슬롯 관리
# ─────────────────────────────────────────────────────────────────────────────

def can_add_position(agent: str) -> bool:
    """에이전트별 슬롯 가용 여부 확인"""
    positions = _load(POS_PATH, {})
    limit = REV_SLOTS if agent == "reversion" else TR_SLOTS
    count = sum(1 for p in positions.values() if p.get("agent") == agent)
    return count < limit


# ─────────────────────────────────────────────────────────────────────────────
# P1-1. 신호 기록 — log_paper_signal()
# ─────────────────────────────────────────────────────────────────────────────

def log_paper_signal(
    ticker: str,
    name: str,
    signal: Any = None,           # 하위 호환용 (미사용)
    entry_price: float | None = None,
    agent: str = "reversion",     # "reversion" | "trend"
    # 확장 파라미터 (선택)
    trigger_types: list[str] | None = None,
    win_prob: float = 0.0,
    avg_win: float = 0.0,
    avg_loss: float = 0.0,
    rr: float = 0.0,
    regime_prob: float | None = None,
    regime_pass: bool = True,
    actual_price: float | None = None,
    position_size_pct: float = 0.0,
    kelly_fraction: float | None = None,
    auc_at_signal: float | None = None,
    eod_close: float | None = None,
    atr_at_entry: float = 0.0,
) -> str:
    """
    신호 기록. signal_id 반환.

    entry_price=None 전달 시 2단계 체결 구조:
      포지션 entry_price=None 저장 → update_entry_prices() 호출 후 실제 Open으로 확정.
      확정 전까지 evaluate_positions()에서 청산 체크 스킵.

    agent="reversion" | "trend" — 슬롯 분리 10+10 관리에 사용.
    """
    # ── 슬롯 방어: 초과 진입 및 동일 종목 중복 진입 차단 ─────────────
    positions = _load(POS_PATH, {})
    if not can_add_position(agent):
        logger.warning("[Paper] 슬롯 초과 — 진입 거부: %s agent=%s", ticker, agent)
        return ""
    held_tickers = {p["ticker"] for p in positions.values()}
    if ticker in held_tickers:
        logger.warning("[Paper] 동일 종목 중복 진입 거부: %s", ticker)
        return ""

    signal_id  = str(uuid.uuid4())[:8]
    now        = _now_kst()
    _eod_close = eod_close if eod_close else entry_price  # fallback
    _triggers  = trigger_types or []

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
        "trigger_types":              _triggers,
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

    # 포지션 추가 (방어 체크에서 이미 로드된 positions 재사용 — 재로드 불필요)
    positions[signal_id] = {
        "signal_id":   signal_id,
        "ticker":      ticker,
        "name":        name,
        "agent":       agent,
        "entry_price":  entry_price,   # None 허용 — update_entry_prices()로 확정
        "eod_close":    _eod_close,
        "entry_date":   now,
        "trade_days":   0,
        "highest":      entry_price or 0.0,
        "atr_at_entry": atr_at_entry,  # trend trailing stop용
    }
    _save(POS_PATH, positions)

    # 시작일 기록 + 파라미터 스냅샷 (최초 신호 시 자동)
    meta = _load(META_PATH, {})
    if "start_date" not in meta:
        meta["start_date"] = _today_kst()
        _save(META_PATH, meta)
        snapshot_params()   # 최초 신호 시 파라미터 동결

    logger.info("[Paper] 신호 기록: %s %s agent=%s (슬롯)", signal_id, ticker, agent)
    return signal_id


# ─────────────────────────────────────────────────────────────────────────────
# P1-2. 청산 평가 — evaluate_positions()
# 매 거래일 장 종료 후 호출. Triple-Barrier와 동일한 청산 로직.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_positions(price_map: dict[str, float], trade_day: bool = True,
                       df_map: dict = None) -> list[dict]:
    """
    price_map: {ticker: 현재가}
    trade_day: 거래일 1일 경과로 카운트할지 여부 (기본 True)
    df_map:    {ticker: DataFrame} — trend MA20/ADX 청산에 필요 (없으면 trailing만 사용)
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

        entry = pos.get("entry_price")
        if entry is None:
            continue  # 시초가 미확정 — trade_days 증가 없음, update_entry_prices() 대기

        if trade_day:
            pos["trade_days"] += 1
        pos["highest"] = max(pos["highest"], cur)
        raw   = (cur - entry) / entry

        reason = None
        agent  = pos.get("agent", "reversion")
        if agent == "trend":
            # trend: trailing 2×ATR → MA20 이탈 → ADX<20 → 60거래일 만기
            atr_entry  = pos.get("atr_at_entry", 0.0)
            trail_stop = pos["highest"] - 2.0 * atr_entry if atr_entry > 0 else entry * 0.85
            if cur < trail_stop:
                reason = "trail"
                exit_price = cur
            else:
                df = (df_map or {}).get(ticker)
                if df is not None and len(df) >= 1:
                    last_row = df.iloc[-1]
                    ma20 = float(last_row["ma20"]) if "ma20" in df.columns else None
                    adx  = float(last_row["adx"])  if "adx"  in df.columns else None
                    if ma20 is not None and cur < ma20:
                        reason = "ma20"
                        exit_price = cur
                    elif adx is not None and adx < 20:
                        reason = "adx"
                        exit_price = cur
            if reason is None and pos["trade_days"] >= 60:
                reason = "time"
                exit_price = cur
        else:
            # reversion: SL -8% 우선 / TP +15% / 10거래일 만기
            if raw <= -SL_PCT:
                reason = "SL"
                exit_price = entry * (1 - SL_PCT)
            elif raw >= TP_PCT:
                reason = "TP"
                exit_price = entry * (1 + TP_PCT)
            elif pos["trade_days"] >= MAX_HOLD_DAYS:
                reason = "time"
                exit_price = cur

        if reason:
            raw_pnl = (exit_price - entry) / entry
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
                "agent":     pos.get("agent", "reversion"),
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
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_positions_auto(trade_day: bool = False) -> list[dict]:
    """
    paper_positions.json의 open 포지션을 실시간 현재가로 자동 평가·청산.
    장중: yfinance fast_info.last_price (실시간), 장외: 최근 종가 fallback.
    trade_day=False (기본): 장중 5분 주기 호출 — trade_days 증가 없음, TP/SL만 체크.
    trade_day=True: EOD 15:30 호출 — 하루치 trade_days 증가.
    """
    positions = _load(POS_PATH, {})
    if not positions:
        return []

    tickers   = list({pos["ticker"] for pos in positions.values()})
    price_map: dict[str, float] = {}

    import yfinance as yf
    from config import KIS_APP_KEY
    _kis = None
    if KIS_APP_KEY:
        try:
            from trader import KISTrader
            _kis = KISTrader()
        except Exception:
            pass
    for ticker in tickers:
        price = 0.0
        try:
            if _kis:
                if ticker.endswith(".KS") or ticker.endswith(".KQ"):
                    code = ticker.replace(".KS", "").replace(".KQ", "")
                    price = float(_kis.get_current_price(code)["price"])
        except Exception:
            pass
        if not price:
            try:
                price = float(yf.Ticker(ticker).fast_info.last_price or 0)
            except Exception:
                pass
        if price > 0:
            price_map[ticker] = price

    if not price_map:
        logger.warning("[Paper] 현재가 조회 실패 — 평가 스킵")
        return []

    closed = evaluate_positions(price_map, trade_day=trade_day)
    if closed:
        try:
            from notifier import send_telegram
            for c in closed:
                send_telegram(
                    f"📋 <b>[페이퍼 청산]</b> {c['name']}({c['ticker']})\n"
                    f"[{c.get('agent','?')}] 사유: {c['reason']} | 세후 {c['net_pnl']:+.3f}% | {c['days']}일 보유"
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
    """누적 측정 지표 반환. market='KR'|None(전체)."""
    trades  = _load(TRADES_PATH, [])
    meta    = _load(META_PATH, {})
    start_date = meta.get("start_date")
    closed  = [t for t in trades
               if t["status"] == "closed"
               and (not start_date or t.get("timestamp", "") >= start_date)]
    if market == "KR":
        closed = [t for t in closed if _is_kr(t["ticker"])]

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

    # 백테스트 갭
    bt_gap = (ev - BACKTEST_EV) if BACKTEST_EV is not None else None

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
    Circuit Breaker 조건 확인.
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
    """Circuit Breaker 상태 확인 (신호 발생 전 게이트)."""
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

def _agent_slot_summary(positions: dict) -> list[str]:
    """에이전트별 슬롯 현황 요약 라인 생성."""
    rev_count = sum(1 for p in positions.values() if p.get("agent") == "reversion")
    tr_count  = sum(1 for p in positions.values() if p.get("agent") == "trend")
    return [
        f"  [reversion] {rev_count}건 / 슬롯 {rev_count}/{REV_SLOTS}",
        f"  [trend]     {tr_count}건 / 슬롯 {tr_count}/{TR_SLOTS}",
    ]


def _market_metrics_section(trades: list, m: dict, backtest_ev_val: float | None) -> list[str]:
    """KR 누적 지표 섹션 생성."""
    lines: list[str] = [
        "",
        "<b>KR 누적 지표</b>",
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

    # 청산 유형별 비율
    closed_kr = [t for t in trades if t.get("status") == "closed" and _is_kr(t["ticker"])]
    if closed_kr:
        n_all = len(closed_kr)
        lines.append("")
        lines.append("<b>KR 청산 유형</b>")
        for reason, rlabel in [("TP", "TP"), ("SL", "SL"), ("time", "기간만료")]:
            subset = [t for t in closed_kr if t.get("exit_reason") == reason]
            cnt = len(subset)
            pct = cnt / n_all * 100
            if reason == "time" and subset:
                avg_net = sum(t["net_pnl_pct"] for t in subset) / len(subset)
                flag = "✅" if avg_net >= 1.0 else ("⚠️" if avg_net >= 0.0 else "❌")
                lines.append(f"  {rlabel}: {cnt}건 ({pct:.0f}%)  평균net={avg_net:+.3f}% {flag}")
            else:
                lines.append(f"  {rlabel}: {cnt}건 ({pct:.0f}%)")

    # 진행률
    elapsed  = m.get("elapsed_days") or 0
    lines.append("")
    lines.append(f"  경과 {elapsed}일 / 14일  |  거래 {m['n']}건 / 50건 목표")

    # 백테스트 비교
    if backtest_ev_val is not None:
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
    market='KR' → KR 섹션, None → KR 전체.
    """
    trades    = _load(TRADES_PATH, [])
    positions = _load(POS_PATH, {})
    today     = _today_kst()

    today_signals   = [t for t in trades if t["timestamp"].startswith(today)]
    today_closed    = [t for t in trades
                       if t["status"] == "closed" and
                       t.get("exit_timestamp", "").startswith(today)]

    mkt_label = "KR"

    lines = [
        f"📋 <b>[페이퍼/{mkt_label}] 일일 리포트 {today}</b>",
        "",
        f"당일 신호: {len(today_signals)}건",
    ]

    if today_signals:
        for s in today_signals:
            lines.append(
                f"  • [{s.get('agent','?')}] {s['name']}({s['ticker']}) "
                f"레짐={s['regime_pass']}"
            )

    lines.append(f"\n당일 청산: {len(today_closed)}건")
    for c in today_closed:
        icon = "✅" if c["is_win"] else "❌"
        lines.append(
            f"  {icon} [{c.get('agent','?')}] {c['name']}({c['ticker']}) "
            f"{c['exit_reason']} {c['net_pnl_pct']:+.3f}%"
        )

    # 미청산 포지션 슬롯 현황
    lines.append(f"\n미청산 포지션: {len(positions)}건")
    lines.extend(_agent_slot_summary(positions))

    for pos in positions.values():
        entry = pos.get("entry_price")
        agent_tag = pos.get("agent", "?")
        if entry:
            try:
                from config import KIS_APP_KEY
                cur = 0.0
                if KIS_APP_KEY:
                    from trader import KISTrader
                    _kt = KISTrader()
                    _tk = pos["ticker"]
                    if _tk.endswith(".KS") or _tk.endswith(".KQ"):
                        cur = float(_kt.get_current_price(_tk.replace(".KS","").replace(".KQ",""))["price"])
                if not cur:
                    import yfinance as yf
                    cur = yf.Ticker(pos["ticker"]).fast_info.last_price or 0
                pnl = (cur - entry) / entry * 100
                arrow = "▲" if pnl >= 0 else "▼"
                lines.append(
                    f"  • [{agent_tag}] {pos['name']}({pos['ticker']}) "
                    f"진입 {entry:,.0f}원 → 현재 {cur:,.0f}원 "
                    f"{arrow}{abs(pnl):.2f}% ({pos['trade_days']}일)"
                )
            except Exception:
                lines.append(
                    f"  • [{agent_tag}] {pos['name']}({pos['ticker']}) "
                    f"진입 {entry:,.0f}원 ({pos['trade_days']}일 경과)"
                )
        else:
            lines.append(
                f"  • [{agent_tag}] {pos['name']}({pos['ticker']}) "
                f"시초가 미확정 — 내일 진입 예정"
            )

    # KR 누적 지표
    m_kr = get_metrics(market="KR")
    lines.extend(_market_metrics_section(trades, m_kr, BACKTEST_EV))
    cb_kr, cb_kr_reason = check_circuit_breaker(market="KR")
    if cb_kr:
        lines += ["", f"🚨 <b>KR Circuit Breaker 발동</b>: {cb_kr_reason}"]
    else:
        lines.append("\n✅ KR Circuit Breaker: 정상")

    # 파라미터 drift 체크
    drifts = check_param_drift()
    if drifts:
        lines += ["", "<b>⚠️ 파라미터 변경 감지 (카운트 리셋 필요)</b>"]
        lines.extend(f"  {d}" for d in drifts)

    # 로직 변경 이력
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
        "REV_SLOTS":        REV_SLOTS,
        "TR_SLOTS":         TR_SLOTS,
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
        if key in ("snapshot_date", "BACKTEST_EV", "REV_SLOTS", "TR_SLOTS"):
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
    KR: 09:05 KST 이후 호출.
    entry_price=None인 포지션에 실제 시초가(Open)를 기록.
    반환: 업데이트된 종목 요약 목록.
    """
    positions = _load(POS_PATH, {})
    trades    = _load(TRADES_PATH, [])

    pending = {
        k: v for k, v in positions.items()
        if v.get("entry_price") is None
        and _is_kr(v["ticker"])
    }
    if not pending:
        return []

    tickers   = list({pos["ticker"] for pos in pending.values()})
    open_map: dict[str, float] = {}

    # ── 1순위: KIS API (당일 시가 stck_oprc) ─────────────────────────────────
    if market == "KR":
        try:
            from config import KIS_APP_KEY
            from trader import KISTrader
            if KIS_APP_KEY:
                t = KISTrader()
                for ticker in tickers:
                    try:
                        code = ticker.replace(".KS", "").replace(".KQ", "")
                        info = t.get_current_price(code)
                        val  = float(info.get("open", 0))
                        if val > 0:
                            open_map[ticker] = val
                            logger.info("[Paper] KIS 시초가: %s open=%s", ticker, val)
                    except Exception as e:
                        logger.warning("[Paper] KIS 시초가 조회 실패 %s: %s", ticker, e)
        except Exception as e:
            logger.warning("[Paper] KIS 초기화 실패, FDR 폴백: %s", e)

    # ── 2순위: FDR ────────────────────────────────────────────────────────────
    remaining = [t for t in tickers if t not in open_map]
    if remaining:
        try:
            import FinanceDataReader as fdr
            from datetime import date
            today = date.today().strftime("%Y-%m-%d")
            for ticker in remaining:
                try:
                    code = ticker.replace(".KS", "").replace(".KQ", "")
                    df = fdr.DataReader(code, today)
                    if not df.empty and "Open" in df.columns:
                        val = float(df["Open"].iloc[-1])
                        if val > 0:
                            open_map[ticker] = val
                            logger.info("[Paper] FDR 시초가: %s open=%s", ticker, val)
                except Exception:
                    pass
        except ImportError:
            pass

    # ── 3순위: yfinance ───────────────────────────────────────────────────────
    remaining = [t for t in tickers if t not in open_map]
    if remaining:
        try:
            import yfinance as yf
            for ticker in remaining:
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
                            logger.info("[Paper] yfinance 시초가: %s open=%s", ticker, val)
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

    # 슬롯 현황
    positions = _load(POS_PATH, {})
    lines.append("")
    lines.extend(_agent_slot_summary(positions))

    try:
        from notifier import send_telegram
        send_telegram("\n".join(l for l in lines if l))
    except Exception:
        pass

    return "\n".join(l for l in lines if l)
