"""
notifier.py - 텔레그램 알림 전송
"""

import requests
import logging
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


def send_telegram(message: str) -> bool:
    """
    텔레그램 봇으로 메시지 전송.
    성공 시 True, 실패 시 False 반환.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("텔레그램 토큰 또는 채팅 ID가 설정되지 않았습니다.")
        print(f"[텔레그램 미설정] 메시지:\n{message}")
        return False

    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML",
    }

    try:
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            logger.info("텔레그램 알림 전송 성공")
            return True
        else:
            logger.error(f"텔레그램 전송 실패: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        logger.error(f"텔레그램 전송 오류: {e}")
        return False


def build_buy_message(stock_name: str, signal: dict, news_summary: str, ai_reason: str) -> str:
    """매수 알림 메시지 포맷 생성"""
    return (
        f"🟢 <b>매수 신호: {stock_name}</b>\n"
        f"5/20 골든크로스 발생 | RSI {signal['rsi']}\n"
        f"📰 뉴스: {news_summary}\n"
        f"🤖 AI 판단: {ai_reason}"
    )


def build_sell_full_message(stock_name: str, signal: dict, ai_reason: str) -> str:
    """전량 매도 알림 메시지 포맷 생성"""
    return (
        f"🔴 <b>전량 매도 신호: {stock_name}</b>\n"
        f"5/20 데드크로스 발생 | RSI {signal['rsi']}\n"
        f"🤖 AI 판단: {ai_reason}"
    )


def build_sell_partial_message(stock_name: str, signal: dict, ai_reason: str) -> str:
    """분할 매도 알림 메시지 포맷 생성"""
    return (
        f"🟡 <b>분할 매도 신호 (50%): {stock_name}</b>\n"
        f"RSI 과매수 후 하락 | RSI {signal['rsi']}\n"
        f"🤖 AI 판단: {ai_reason}"
    )


def build_warning_message(stock_name: str, consecutive_count: int) -> str:
    """연속 매수 신호 경고 메시지 생성"""
    return (
        f"⚠️ <b>연속 매수 경고: {stock_name}</b>\n"
        f"연속 {consecutive_count}회 매수 신호 발생 — 과열 주의"
    )
