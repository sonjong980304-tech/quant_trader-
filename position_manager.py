"""
position_manager.py - ML 포지션 추적 및 봇 활성화 상태 관리

- bot_active 상태 파일 (state.json) 읽기/쓰기
- ML 포지션 저장 / 익절 / 손절 / 7거래일 자동 청산
- 봇 활성화 게이트 (_check_activation)
"""

import json as _json
import logging
from datetime import datetime, timedelta

import yfinance as yf
import pytz

from config import KIS_APP_KEY, LIVE_TRADING, SL_PCT
from trader import KISTrader
from notifier import send_telegram
from market_calendar import is_kr_trading_day

KST = pytz.timezone("Asia/Seoul")
logger = logging.getLogger(__name__)


def _is_market_open(is_us: bool) -> bool:
    """현재 시각이 해당 시장 장중 시간인지 반환 (테스트에서 monkeypatch 가능)."""
    now_kst = datetime.now(KST)
    if is_us:
        eastern = pytz.timezone("America/New_York")
        now_et  = now_kst.astimezone(eastern)
        et_min  = now_et.hour * 60 + now_et.minute
        return (now_et.weekday() < 5) and (9 * 60 + 30 <= et_min < 16 * 60)
    kr_min = now_kst.hour * 60 + now_kst.minute
    return is_kr_trading_day(now_kst.date()) and (9 * 60 <= kr_min < 15 * 60 + 30)

_STATE_FILE = "/Users/gyuyeong/quant_trader/state.json"


def _load_state() -> dict:
    try:
        with open(_STATE_FILE) as f:
            return _json.load(f)
    except Exception:
        return {"bot_active": False, "legacy_tickers": [], "activated_at": None}


def _save_state(state: dict):
    with open(_STATE_FILE, "w") as f:
        _json.dump(state, f, ensure_ascii=False, indent=2)


def _init_legacy_tickers():
    """최초 실행 시 현재 보유 종목을 legacy_tickers로 기록."""
    state = _load_state()
    if state.get("bot_active") or state.get("legacy_tickers"):
        return

    if not KIS_APP_KEY:
        return

    try:
        balance = KISTrader().get_balance()
        tickers = [h["stock_code"] for h in balance if h.get("qty", 0) > 0]
        if tickers:
            state["legacy_tickers"] = tickers
            _save_state(state)
            logger.info("기존 보유 종목 기록: %s — 매도 완료 시 봇 자동 활성화", tickers)
            send_telegram(
                f"⏸ <b>봇 대기 중</b>\n"
                f"기존 보유 종목 {tickers}을 매도하면 자동 시작됩니다.\n"
                f"현재 텔레그램 LLM 기능은 정상 작동합니다."
            )
    except Exception as e:
        logger.warning("legacy_tickers 초기화 실패: %s", e)


def _check_activation():
    """
    기존 종목이 모두 매도됐는지 확인.
    모두 사라지면 bot_active = True 로 전환하고 텔레그램 알림.
    """
    state = _load_state()
    if state.get("bot_active"):
        return True

    if state.get("user_stopped"):
        return False

    legacy = state.get("legacy_tickers", [])
    if not legacy:
        state["bot_active"] = True
        state["activated_at"] = datetime.now(KST).isoformat()
        _save_state(state)
        return True

    if not KIS_APP_KEY:
        return False

    try:
        balance       = KISTrader().get_balance()
        held          = {h["stock_code"] for h in balance if h.get("qty", 0) > 0}
        still_holding = [t for t in legacy if t in held]

        if not still_holding:
            state["bot_active"]     = True
            state["activated_at"]   = datetime.now(KST).isoformat()
            state["legacy_tickers"] = []
            _save_state(state)
            logger.info("기존 종목 전량 매도 확인 → 봇 활성화!")
            send_telegram(
                "✅ <b>퀀트 봇 활성화!</b>\n"
                "기존 보유 종목이 모두 매도됐습니다.\n"
                "안전자산 포트폴리오 및 급등주 ML 전략을 시작합니다."
            )
            return True
        else:
            logger.info("봇 대기 중 — 아직 보유: %s", still_holding)
            return False
    except Exception as e:
        logger.warning("활성화 체크 실패: %s", e)
        return False


def is_bot_active() -> bool:
    return _load_state().get("bot_active", False)


# ─────────────────────────────────────────────
# ML 포지션 추적 (익절 / 7거래일 자동 청산)
# ─────────────────────────────────────────────

def save_ml_position(ticker: str, name: str, qty: int, entry_price: float,
                     avg_win: float, atr: float = 0.0, is_us: bool = False):
    """매수 확정 시 ML 포지션 저장."""
    state    = _load_state()
    positions = state.setdefault("ml_positions", {})
    target_price = entry_price * (1 + avg_win)
    # ATR 기반 손절: ATR이 가격의 10% 초과 시 데이터 이상으로 간주 → SL_PCT 고정 폴백
    # 정상 ATR이라도 stop은 최대 -SL_PCT 이하로 내려가지 않도록 캡
    if atr > 0 and (atr / entry_price) < 0.10:
        stop_price = max(entry_price - 2.0 * atr, entry_price * (1 - SL_PCT))
    else:
        stop_price = entry_price * (1 - SL_PCT)
    positions[ticker] = {
        "ticker":        ticker,
        "name":          name,
        "qty":           qty,
        "entry_price":   round(entry_price, 4),
        "target_price":  round(target_price, 4),
        "stop_price":    round(stop_price, 4),
        "entry_date":    datetime.now(KST).strftime("%Y-%m-%d"),
        "is_us":         is_us,
        "atr":           round(atr, 4),
        "highest_price": round(entry_price, 4),
    }
    _save_state(state)
    logger.info("ML 포지션 저장: %s qty=%d entry=%.2f target=%.2f stop=%.2f atr=%.4f",
                ticker, qty, entry_price, target_price, stop_price, atr)


def _get_current_price(ticker: str):
    """현재가 조회 — KR: KIS API, US: KIS API 우선 → yfinance fallback."""
    if KIS_APP_KEY:
        try:
            t = KISTrader()
            if ticker.endswith(".KS") or ticker.endswith(".KQ"):
                code = ticker.replace(".KS", "").replace(".KQ", "")
                return float(t.get_current_price(code)["price"])
            else:
                return float(t.get_us_current_price(ticker)["price"])
        except Exception:
            pass
    try:
        return float(yf.Ticker(ticker).fast_info["lastPrice"])
    except Exception:
        return None


def _trading_days_elapsed(entry_date_str: str) -> int:
    """입력 날짜부터 오늘까지 KRX 거래일(공휴일 포함) 수."""
    from datetime import date as date_cls
    start = date_cls.fromisoformat(entry_date_str)
    today = date_cls.today()
    days  = 0
    d     = start
    while d < today:
        d += timedelta(days=1)
        if is_kr_trading_day(d):
            days += 1
    return days


def check_ml_positions():
    """
    5분마다 실행 — ML 포지션 익절 / 손절 / 7거래일 자동 청산 체크.
    - 현재가 >= target_price  → 익절 매도
    - 현재가 <= stop_price    → 손절 매도
    - 7거래일 경과             → 강제 청산
    """
    # KST 주말(토/일)에는 기본 스킵 — 단, 일요일 밤 KST = 미국 월요일 장 시작이므로
    # US 장이 열려있으면 계속 실행 (일요일 밤 KST에 US 포지션 TP/SL 체크 필요)
    now_kst = datetime.now(KST)
    if now_kst.weekday() >= 5:
        import pytz as _pytz
        _eastern = _pytz.timezone("America/New_York")
        _now_et  = now_kst.astimezone(_eastern)
        _et_min  = _now_et.hour * 60 + _now_et.minute
        _us_open = (_now_et.weekday() < 5) and (9 * 60 + 30 <= _et_min < 16 * 60)
        if not _us_open:
            return

    state     = _load_state()
    positions = state.get("ml_positions", {})
    if not positions:
        return

    to_remove = []
    for ticker, pos in positions.items():
        try:
            is_us = pos.get("is_us", False)

            # 장 시간 외에는 TP/SL/트레일링 체크 스킵 (7거래일 경과는 항상 체크)
            in_market = _is_market_open(is_us)

            cur_price = _get_current_price(ticker) if in_market else None
            elapsed   = _trading_days_elapsed(pos["entry_date"])
            qty       = pos["qty"]
            name      = pos.get("name", ticker)
            target    = pos["target_price"]
            stop      = pos.get("stop_price", pos["entry_price"] * 0.93)

            if cur_price and cur_price > pos.get("highest_price", pos["entry_price"]):
                pos["highest_price"] = round(cur_price, 4)
                state["ml_positions"][ticker] = pos
                _save_state(state)

            highest = pos.get("highest_price", pos["entry_price"])
            atr_val = pos.get("atr", 0.0)

            reason = None
            emoji  = "⏰"
            if cur_price and cur_price >= target:
                reason = f"익절 (현재가 {cur_price:.4f} ≥ 목표 {target:.4f})"
                emoji  = "✅"
            elif cur_price and cur_price <= stop:
                reason = f"ATR손절 (현재가 {cur_price:.4f} ≤ 손절가 {stop:.4f})"
                emoji  = "🔴"
            elif cur_price and highest > pos["entry_price"]:
                trail_pct = cur_price < highest * (1 - 0.025)
                trail_atr = atr_val > 0 and cur_price < highest - atr_val
                if trail_pct or trail_atr:
                    reason = f"트레일링스톱 (고점 {highest:.4f} → 현재 {cur_price:.4f})"
                    emoji  = "📉"
            if reason is None and elapsed >= 7:
                reason = f"7거래일 경과 ({elapsed}일)"

            if reason is None:
                continue

            logger.info("[%s] 자동 매도 — %s", ticker, reason)
            sell_price = cur_price or pos["entry_price"]
            if not LIVE_TRADING:
                logger.warning("[LIVE_TRADING=False] 실매도 차단 — 페이퍼: %s %s", ticker, reason)
            elif KIS_APP_KEY:
                t    = KISTrader()
                code = ticker.replace(".KS", "").replace(".KQ", "")
                if is_us:
                    t.sell_us(code, qty)
                else:
                    t.sell(code, qty)

            try:
                from trade_logger import log_sell
                log_sell(ticker, sell_price, qty=qty, notes=reason)
            except Exception as e:
                logger.warning("[TradeLog] 매도 기록 실패 [%s]: %s", ticker, e)

            pnl = (sell_price - pos["entry_price"]) / pos["entry_price"] * 100
            send_telegram(
                f"{emoji} <b>{name} ({ticker})</b>\n"
                f"사유: {reason}\n"
                f"수량: {qty}주 | 손익: {pnl:+.2f}%"
            )
            to_remove.append(ticker)

        except Exception as e:
            logger.error("[%s] 자동 청산 실패: %s", ticker, e)
            if "수량" in str(e) and "초과" in str(e):
                logger.warning("[%s] 실제 미보유 확인 — state.json에서 제거", ticker)
                to_remove.append(ticker)

    if to_remove:
        for t in to_remove:
            positions.pop(t, None)
        state["ml_positions"] = positions
        _save_state(state)

    try:
        from paper_trader import evaluate_positions_auto
        evaluate_positions_auto()
    except Exception as _pe:
        logger.debug("[Paper] 포지션 평가 오류: %s", _pe)
