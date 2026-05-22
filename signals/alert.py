from __future__ import annotations

"""
alert.py - 급등 신호 텔레그램 알림 전송

메시지 포맷:
  - 트리거 종류
  - ML 예측 승률 / 예상 수익 / 예상 손실 / 손익비
  - 켈리 공식 추천 투자 비중 및 수량
  - 매수 확인 버튼 (/buysignal_XXX) 또는 패스 (/skipsignal)
"""

import logging
from notifier import send_telegram
from portfolio.kelly import kelly_fraction, position_size

logger = logging.getLogger(__name__)


def build_signal_message(signal: dict, growth_cash: float) -> str:
    """
    급등 신호 텔레그램 메시지 생성.

    signal      : scanner.scan_ticker() 반환값
    growth_cash : 급등주 전용 가용 현금 (전체 자산 × 30%)
    """
    ticker    = signal["ticker"]
    name      = signal.get("name", ticker)
    win_prob  = signal["win_prob"] * 100
    avg_win   = signal["avg_win"]  * 100
    avg_loss  = signal["avg_loss"] * 100
    rr        = signal["risk_reward"]
    price     = signal["current_price"]
    triggers  = ", ".join(signal["triggers"])

    kelly_f = kelly_fraction(signal["win_prob"], signal["avg_win"], signal["avg_loss"])
    qty, _, invest_amount = position_size(
        growth_cash, signal["win_prob"], signal["avg_win"], signal["avg_loss"], price
    )

    # 텔레그램 커맨드용 티커 (점 → 언더스코어)
    safe_ticker = ticker.replace(".", "_")

    return (
        f"🚨 <b>[급등 신호] {name} ({ticker})</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📡 트리거: {triggers}\n"
        f"💵 현재가: {price:,.0f}원\n\n"
        f"📊 <b>ML 예측 (7일 기준)</b>\n"
        f"  승률:        <b>{win_prob:.1f}%</b>\n"
        f"  예상 수익:   <b>+{avg_win:.1f}%</b> (성공 시)\n"
        f"  예상 손실:   <b>-{avg_loss:.1f}%</b> (실패 시)\n"
        f"  손익비:      <b>{rr:.2f}</b>\n\n"
        f"💰 <b>켈리 추천 투자</b>\n"
        f"  비중:  현금의 {kelly_f*100:.1f}%\n"
        f"  금액:  {invest_amount:,.0f}원\n"
        f"  수량:  {qty}주\n\n"
        f"매수하시겠습니까?\n"
        f"✅ /buysignal_{safe_ticker} — 매수\n"
        f"❌ /skipsignal — 패스"
    )


def send_signal_alert(signal: dict, growth_cash: float) -> bool:
    """급등 신호 알림 전송."""
    try:
        msg = build_signal_message(signal, growth_cash)
        return send_telegram(msg)
    except Exception as e:
        logger.error("신호 알림 전송 실패: %s", e)
        return False


def build_buy_confirm_message(signal: dict, qty: int, price: float) -> str:
    """매수 확정 후 전송할 확인 메시지."""
    name = signal.get("name", signal["ticker"])
    return (
        f"✅ <b>급등주 매수 완료</b>\n"
        f"종목: {name} ({signal['ticker']})\n"
        f"수량: {qty}주\n"
        f"가격: {price:,.0f}원\n"
        f"금액: {qty * price:,.0f}원\n\n"
        f"📌 7일 내 +{signal['avg_win']*100:.1f}% 목표"
    )
