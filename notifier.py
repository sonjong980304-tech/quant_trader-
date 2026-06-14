"""
notifier.py - 텔레그램 알림 전송
"""

import json
import logging
import requests

from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    MA_LONG, MA20_RISING_LOOKBACK, VOLUME_LOOKBACK_DAYS,
)

logger = logging.getLogger(__name__)


def send_telegram(message: str) -> bool:
    """텔레그램 봇으로 메시지 전송. 성공 시 True, 실패 시 False."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("텔레그램 토큰 또는 채팅 ID가 설정되지 않았습니다.")
        print(f"[텔레그램 미설정] 메시지:\n{message}")
        return False

    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}

    try:
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            logger.info("텔레그램 알림 전송 성공")
            return True
        logger.error("텔레그램 전송 실패: %s %s", resp.status_code, resp.text)
        return False
    except Exception as e:
        logger.error("텔레그램 전송 오류: %s", e)
        return False


def send_buy_confirmation_keyboard(text: str, conf_id: str) -> bool:
    """EOD 매수 신호 확인 메시지를 인라인 키보드(✅/❌)와 함께 전송."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("텔레그램 토큰 또는 채팅 ID가 설정되지 않았습니다.")
        return False

    url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":      TELEGRAM_CHAT_ID,
        "text":         text,
        "parse_mode":   "HTML",
        "reply_markup": json.dumps({
            "inline_keyboard": [[
                {"text": "✅ 매수 확인", "callback_data": f"buy_confirm_{conf_id}"},
                {"text": "❌ 취소",      "callback_data": f"buy_cancel_{conf_id}"},
            ]]
        }),
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("매수 확인 키보드 전송 성공 (conf_id=%s)", conf_id)
            return True
        logger.error("키보드 전송 실패: %s %s", resp.status_code, resp.text)
        return False
    except Exception as e:
        logger.error("키보드 전송 오류: %s", e)
        return False


# ─────────────────────────────────────────────
# 매수 / 매도 신호 메시지
# ─────────────────────────────────────────────

def build_buy_message(stock_name: str, signal: dict, buy_principle: str = "") -> str:
    principles = "/".join(signal.get("buy_which", [])) or buy_principle or "?"
    return (
        f"🟢 <b>매수 신호: {stock_name}</b>\n"
        f"원칙: {principles} | 현재가: {signal.get('close', 0):,.0f}원\n"
        f"MA5: {signal['ma_short']:,.0f} | MA20: {signal['ma_long']:,.0f} | RSI: {signal['rsi']}"
    )


def build_sell_full_message(stock_name: str, signal: dict, reason: str = "") -> str:
    reason_line = f"\n사유: {reason}" if reason else ""
    return (
        f"🔴 <b>매도 신호 (전량): {stock_name}</b>\n"
        f"현재가: {signal.get('close', 0):,.0f}원\n"
        f"MA5: {signal['ma_short']:,.0f} | MA20: {signal['ma_long']:,.0f}"
        f"{reason_line}"
    )


def build_sell_partial_message(stock_name: str, signal: dict, reason: str = "") -> str:
    reason_line = f"\n사유: {reason}" if reason else ""
    return (
        f"🟡 <b>매도 신호 (1원칙-부분): {stock_name}</b>\n"
        f"5일선 위 거래량 급증 + 장대음봉 | 현재가: {signal.get('close', 0):,.0f}원\n"
        f"MA5: {signal['ma_short']:,.0f} | MA20: {signal['ma_long']:,.0f}"
        f"{reason_line}"
    )


def build_daily_summary_message(
    stock_name: str,
    sig: dict,
    daily_df,
    position: dict = None,
) -> str:
    """오후 3시 일일 기술적 분석 리포트 — 종목 1건 메시지"""
    close  = sig["close"]
    ma5    = sig["ma_short"]
    ma20   = sig["ma_long"]
    rsi    = sig["rsi"]
    volume = sig["volume"]

    # 전일 대비 등락
    prev_close = float(daily_df["Close"].iloc[-2]) if len(daily_df) >= 2 else close
    change_pct = (close - prev_close) / prev_close * 100 if prev_close > 0 else 0
    change_icon = "▲" if change_pct >= 0 else "▼"

    # 현재가 vs MA 위치
    pos_ma5  = "위" if close > ma5  else "아래"
    pos_ma20 = "위" if close > ma20 else "아래"

    # MA20 방향 (MA20_RISING_LOOKBACK일 전과 비교)
    ma20_col = f"MA_{MA_LONG}"
    if ma20_col in daily_df.columns and len(daily_df) > MA20_RISING_LOOKBACK:
        ma20_rising = float(daily_df[ma20_col].iloc[-1]) > float(daily_df[ma20_col].iloc[-1 - MA20_RISING_LOOKBACK])
    else:
        ma20_rising = False
    ma20_dir = "우상향↗" if ma20_rising else "우하향↘"

    # 거래량 vs 50일 평균
    vol_avg = float(
        daily_df["Volume"].rolling(window=VOLUME_LOOKBACK_DAYS, min_periods=1).mean().shift(1).iloc[-1]
    )
    vol_ratio = volume / vol_avg if vol_avg > 0 else 0

    # 캔들 타입
    open_price = float(daily_df["Open"].iloc[-1])
    if close > open_price:
        candle = "양봉"
    elif close < open_price:
        candle = "음봉"
    else:
        candle = "도지"

    # 매수/매도 신호
    buy_which  = sig.get("buy_which", [])
    sell_which = sig.get("sell_which", [])
    if buy_which:
        signal_str = f"🟢 매수 ({'/'.join(buy_which)})"
    elif sell_which:
        signal_str = f"🔴 매도 ({'/'.join(sell_which)})"
    else:
        signal_str = "─ 없음"

    # 보유 현황
    if position:
        avg_price  = position.get("avg_price") or position.get("entry_price", 0)
        profit_pct = (close - avg_price) / avg_price * 100 if avg_price > 0 else 0
        p_icon     = "▲" if profit_pct >= 0 else "▼"
        hold_str   = f"보유중 | 매수가 {avg_price:,.0f}원 | 수익률 {p_icon}{abs(profit_pct):.1f}%"
    else:
        hold_str = "미보유"

    return (
        f"<b>📈 {stock_name}</b>\n"
        f"현재가: {close:,.0f}원  {change_icon}{abs(change_pct):.2f}%\n"
        f"MA5: {ma5:,.0f} | MA20: {ma20:,.0f} ({ma20_dir})\n"
        f"위치: MA5 {pos_ma5} / MA20 {pos_ma20}\n"
        f"RSI: {rsi} | 캔들: {candle}\n"
        f"거래량: {vol_ratio:.1f}배 ({VOLUME_LOOKBACK_DAYS}일 평균 대비)\n"
        f"신호: {signal_str}\n"
        f"포지션: {hold_str}"
    )


def build_warning_message(stock_name: str, consecutive_count: int) -> str:
    return (
        f"⚠️ <b>연속 매수 경고: {stock_name}</b>\n"
        f"연속 {consecutive_count}회 매수 신호 발생 — 과열 주의"
    )


# ─────────────────────────────────────────────
# 익절 알림
# ─────────────────────────────────────────────

def send_take_profit_alert(stock_name: str, price: float, profit_type: str) -> bool:
    if profit_type == "half":
        msg = f"✅ [1차익절] {stock_name} {price:,.0f}원 | +8% 달성 | 50% 매도"
    else:
        msg = f"🎯 [2차익절] {stock_name} {price:,.0f}원 | +15% 달성 | 전량 매도"
    return send_telegram(msg)


# ─────────────────────────────────────────────
# 분봉 거래량 급증 알림
# ─────────────────────────────────────────────

def build_volume_surge_message(
    stock_name: str,
    current_price: float,
    surge_ratio: float,
    news_items: list,
) -> str:
    """분봉 거래량 급증 + 최신 뉴스 3건 알림 메시지"""
    lines = [
        f"⚡ <b>거래량 급증: {stock_name}</b>",
        f"현재가: {current_price:,.0f}원 | 분봉 거래량 평균 대비 {surge_ratio:.1f}배",
    ]
    if news_items:
        lines.append("📰 최신 뉴스")
        for i, item in enumerate(news_items, 1):
            title = item.get("title", "제목 없음")
            link  = item.get("link", "")
            lines.append(f"{i}. <a href=\"{link}\">{title}</a>")
    else:
        lines.append("📰 관련 뉴스 없음")
    return "\n".join(lines)
