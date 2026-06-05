from __future__ import annotations

"""
alert.py - 급등 신호 자동 매수 실행 + 텔레그램 알림

흐름:
  1. 켈리 공식으로 매수 수량 계산 (US 주식은 USD→KRW 환산 후 수량 계산)
  2. qty == 0이면 주문가능금액으로 최대 매수 재시도
  3. qty > 0이면 KIS API로 즉시 매수 실행
  4. 결과를 텔레그램으로 전송
"""

import logging
from notifier import send_telegram
from portfolio.kelly import kelly_fraction, position_size

logger = logging.getLogger(__name__)

_EXCHANGE_RATE = 1_400   # 원/달러 근사치 (통합증거금 기준)


def _is_us(ticker: str) -> bool:
    return not (ticker.endswith(".KS") or ticker.endswith(".KQ"))


def _price_krw(price: float, ticker: str) -> float:
    """US 주식 달러 가격을 원화 환산 (수량 계산용)."""
    return price * _EXCHANGE_RATE if _is_us(ticker) else price


def _execute_buy(signal: dict, growth_cash: float) -> tuple[int, float, str]:
    """
    매수 실행.
    반환: (qty, price, status)
      status: "ok" | "skip_qty" | "skip_no_key" | "error:<msg>"
    """
    from config import KIS_APP_KEY

    ticker    = signal["ticker"]
    price     = signal["current_price"]
    us        = _is_us(ticker)
    price_for = _price_krw(price, ticker)   # 수량 계산용 원화 환산가

    qty, _, _ = position_size(
        growth_cash, signal["win_prob"], signal["avg_win"], signal["avg_loss"], price_for
    )

    # 켈리 추천 0주 → 주문가능금액으로 최대 매수 재시도
    if qty <= 0 and KIS_APP_KEY:
        try:
            from trader import KISTrader
            avail = float(KISTrader().get_available_cash())
            qty   = int(avail / price_for)
            if qty > 0:
                logger.info("[%s] 켈리 0주 → 주문가능금액으로 %d주 매수 시도", ticker, qty)
        except Exception as e:
            logger.warning("[%s] 주문가능금액 조회 실패: %s", ticker, e)

    if qty <= 0:
        return 0, price, "skip_qty"

    if not KIS_APP_KEY:
        return qty, price, "skip_no_key"

    code = ticker.replace(".KS", "").replace(".KQ", "")

    try:
        from trader import KISTrader
        t = KISTrader()
        if us:
            t.buy_us(code, qty)
        else:
            t.buy(code, qty)

        try:
            from trade_logger import log_buy
            log_buy(
                ticker, signal.get("name", ticker), price, qty,
                strategy  = "ML급등주",
                win_prob  = signal.get("win_prob"),
                avg_win   = signal.get("avg_win"),
                avg_loss  = signal.get("avg_loss"),
                model_auc = signal.get("model_auc"),
            )
        except Exception as _e:
            logger.warning("[TradeLog] 매수 기록 실패 [%s]: %s", ticker, _e)

        return qty, price, "ok"
    except Exception as e:
        logger.error("[%s] 매수 실패: %s", ticker, e)
        return qty, price, f"error:{e}"


def send_signal_alert(signal: dict, growth_cash: float) -> dict:
    """
    급등 신호 자동 매수 실행 후 텔레그램 알림.
    반환: {"sent": bool, "qty": int, "price": float, "status": str}
    """
    ticker      = signal["ticker"]
    name        = signal.get("name", ticker)
    win_prob    = signal["win_prob"] * 100
    avg_win     = signal["avg_win"]  * 100
    avg_loss    = signal["avg_loss"] * 100
    rr          = signal["risk_reward"]
    price       = signal["current_price"]
    triggers    = ", ".join(signal["triggers"])
    us          = _is_us(ticker)
    price_for   = _price_krw(price, ticker)
    agent_label = {"momentum": "돌파 에이전트", "reversion": "눌림목 에이전트"}.get(
        signal.get("agent", ""), "통합 에이전트"
    )

    kelly_f = kelly_fraction(signal["win_prob"], signal["avg_win"], signal["avg_loss"])
    qty, _, invest_amount = position_size(
        growth_cash, signal["win_prob"], signal["avg_win"], signal["avg_loss"], price_for
    )

    # 표시용 가격/금액 문자열
    if us:
        price_str  = f"${price:,.2f}"
        invest_str = f"${invest_amount / _EXCHANGE_RATE:,.0f} (₩{invest_amount:,.0f})"
    else:
        price_str  = f"{price:,.0f}원"
        invest_str = f"{invest_amount:,.0f}원"

    try:
        qty_executed, exec_price, status = _execute_buy(signal, growth_cash)

        if status == "skip_qty":
            exec_line = "⚠️ 켈리 기준 0주 — 1주 가격이 주문가능금액 초과, 매수 생략"
        elif status == "skip_no_key":
            exec_line = f"⚠️ KIS API 미연결 — 실제 매수 생략 (추천 {qty}주)"
        elif status == "ok":
            if us:
                exec_line = (
                    f"✅ <b>자동 매수 완료</b> {qty_executed}주 × ${exec_price:,.2f} "
                    f"= ${qty_executed * exec_price:,.0f}"
                )
            else:
                exec_line = (
                    f"✅ <b>자동 매수 완료</b> {qty_executed}주 × {exec_price:,.0f}원 "
                    f"= {qty_executed * exec_price:,.0f}원"
                )
        else:
            exec_line = f"❌ 매수 실패: {status.replace('error:', '')}"

        msg = (
            f"🚨 <b>[급등 신호] {name} ({ticker})</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🤖 에이전트: <b>{agent_label}</b>\n"
            f"📡 트리거: {triggers}\n"
            f"💵 현재가: {price_str}\n\n"
            f"📊 <b>ML 예측 (7일 기준)</b>\n"
            f"  승률:        <b>{win_prob:.1f}%</b>\n"
            f"  예상 수익:   <b>+{avg_win:.1f}%</b>\n"
            f"  예상 손실:   <b>-{avg_loss:.1f}%</b>\n"
            f"  손익비:      <b>{rr:.2f}</b>\n\n"
            f"💰 <b>켈리 추천</b> 비중 {kelly_f*100:.1f}% / {invest_str} / {qty}주\n\n"
            f"{exec_line}"
        )
        sent = send_telegram(msg)
        return {"sent": sent, "qty": qty_executed, "price": exec_price, "status": status}

    except Exception as e:
        logger.error("신호 알림/매수 실패: %s", e)
        return {"sent": False, "qty": 0, "price": signal.get("current_price", 0), "status": f"error:{e}"}


def build_buy_confirm_message(signal: dict, qty: int, price: float) -> str:
    """수동 매수 확정 후 전송할 확인 메시지 (telegram_bot.py /buy 커맨드용)."""
    name = signal.get("name", signal["ticker"])
    return (
        f"✅ <b>수동 매수 완료</b>\n"
        f"종목: {name} ({signal['ticker']})\n"
        f"수량: {qty}주\n"
        f"가격: {price:,.0f}원\n"
        f"금액: {qty * price:,.0f}원"
    )
