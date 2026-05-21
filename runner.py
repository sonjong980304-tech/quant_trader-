"""
runner.py - 장중 우선순위 기반 자동매매 스케줄러
실행 순서 (매 사이클):
  ① 익절 → ② 매도신호 → ③ 매수신호

분봉 거래량 급증 감지 시 네이버 최신 뉴스 3건을 텔레그램으로 전송.
"""

import schedule
import time
import logging
import logging.handlers
from datetime import datetime, timedelta

import pytz

from config import (
    STOCKS, MA_SHORT, MA_LONG, RSI_PERIOD, KIS_APP_KEY,
    VOLUME_SURGE_MINUTE_RATIO,
)
from data_fetcher import fetch_ohlcv, get_minute_data
from indicators import add_all_indicators, detect_crossover
from strategy import generate_signals, get_latest_signal, buy_signal_1_intraday
from trader import (
    positions, KISTrader,
    check_take_profit, calc_position_size,
    register_position, clear_position,
    execute_buy, execute_sell_all, execute_sell_half,
)
from notifier import (
    send_take_profit_alert,
    build_buy_message, build_sell_full_message, build_sell_partial_message,
    send_telegram, build_volume_surge_message, build_daily_summary_message,
)
from news_fetcher import fetch_naver_news

KST      = pytz.timezone("Asia/Seoul")
LOG_FILE = "logs/trader.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("runner")

# 종목별 마지막 거래량 급증 알림 시각 (30분 이내 중복 방지)
_surge_alerted: dict = {}
_SURGE_COOLDOWN_MINUTES = 30


def is_market_hours() -> bool:
    now = datetime.now(KST)
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def _check_volume_surge(ticker: str, stock_name: str, minute_df, current_price: float):
    """
    분봉 거래량 급증 감지 → 네이버 뉴스 3건 텔레그램 전송.
    직전 5분봉 평균 거래량 대비 VOLUME_SURGE_MINUTE_RATIO배 이상이면 급증으로 판단.
    """
    if minute_df is None or minute_df.empty or len(minute_df) < 6:
        return

    current_vol = float(minute_df["volume"].iloc[-1])
    avg_vol     = float(minute_df["volume"].iloc[-6:-1].mean())

    if avg_vol <= 0:
        return

    surge_ratio = current_vol / avg_vol
    if surge_ratio < VOLUME_SURGE_MINUTE_RATIO:
        return

    # 쿨다운 체크 (30분 이내 중복 알림 방지)
    last_alert = _surge_alerted.get(ticker)
    now        = datetime.now(KST)
    if last_alert and (now - last_alert) < timedelta(minutes=_SURGE_COOLDOWN_MINUTES):
        return

    logger.info("  [급증] %s 분봉 거래량 %.1f배 — 뉴스 수집 중", stock_name, surge_ratio)
    _surge_alerted[ticker] = now

    news_items = fetch_naver_news(stock_name, n=3)
    msg = build_volume_surge_message(stock_name, current_price, surge_ratio, news_items)
    send_telegram(msg)


def send_daily_summary():
    """오후 3시 종목별 일일 기술적 분석 리포트를 텔레그램으로 전송."""
    now = datetime.now(KST)
    if now.weekday() >= 5:
        return

    logger.info("일일 기술적 분석 리포트 전송 시작")
    send_telegram(f"📊 <b>일일 기술적 분석 리포트</b>\n{now.strftime('%Y-%m-%d %H:%M')} 기준")

    for ticker, stock_name in STOCKS.items():
        try:
            daily_df = fetch_ohlcv(ticker, period_years=1)
            daily_df = add_all_indicators(daily_df, short=MA_SHORT, long=MA_LONG, rsi_period=RSI_PERIOD)
            daily_df = detect_crossover(daily_df, short=MA_SHORT, long=MA_LONG)
            daily_df = generate_signals(daily_df)
            sig      = get_latest_signal(daily_df)
            msg      = build_daily_summary_message(
                stock_name, sig, daily_df, position=positions.get(ticker)
            )
            send_telegram(msg)
        except Exception as e:
            logger.error("  일일 리포트 실패 (%s): %s", stock_name, e)

    logger.info("일일 기술적 분석 리포트 전송 완료")


def run_priority_loop():
    if not is_market_hours():
        logger.info("장 시간 외 — 실행 건너뜀 (%s)", datetime.now(KST).strftime("%H:%M"))
        return

    now = datetime.now(KST)
    logger.info("=" * 55)
    logger.info("매매 루프 시작 (%s)", now.strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 55)

    if KIS_APP_KEY:
        try:
            trader         = KISTrader()
            available_cash = trader.get_available_cash()
        except Exception as e:
            logger.error("KISTrader 초기화 실패: %s", e)
            return
    else:
        trader         = None
        available_cash = 10_000_000
        logger.info("KIS_APP_KEY 미설정 — 시뮬레이션 모드")

    for ticker, stock_name in STOCKS.items():
        stock_code = ticker.replace(".KS", "").replace(".KQ", "")
        logger.info("\n>>> [%s] %s 처리 시작", ticker, stock_name)

        # ── 일봉 데이터 + 지표 계산 ──────────────────────
        try:
            daily_df = fetch_ohlcv(ticker, period_years=1)
            daily_df = add_all_indicators(
                daily_df, short=MA_SHORT, long=MA_LONG, rsi_period=RSI_PERIOD
            )
            daily_df = detect_crossover(daily_df, short=MA_SHORT, long=MA_LONG)
            daily_df = generate_signals(daily_df)
        except Exception as e:
            logger.error("  일봉 데이터 처리 실패: %s", e)
            continue

        # ── 현재가 / 분봉 데이터 ──────────────────────────
        minute_df = None
        try:
            if trader:
                price_info    = trader.get_current_price(stock_code)
                current_price = float(price_info["price"])
            else:
                current_price = float(daily_df["Close"].iloc[-1])

            try:
                minute_df = get_minute_data(ticker, interval_min=1)
            except Exception as e:
                logger.warning("  분봉 데이터 조회 실패: %s", e)

        except Exception as e:
            logger.error("  현재가 조회 실패: %s", e)
            continue

        # ── 분봉 거래량 급증 감지 ────────────────────────
        if minute_df is not None and not minute_df.empty:
            _check_volume_surge(ticker, stock_name, minute_df, current_price)

        # ① 익절 체크
        profit_type = check_take_profit(ticker, current_price)
        if profit_type == "half":
            logger.info("  1차 익절 (50%%) 실행: %s", ticker)
            execute_sell_half(ticker)
            positions[ticker]["half_sold"] = True
            send_take_profit_alert(stock_name, current_price, "half")
            continue
        if profit_type == "full":
            logger.info("  2차 익절 (전량) 실행: %s", ticker)
            execute_sell_all(ticker)
            send_take_profit_alert(stock_name, current_price, "full")
            clear_position(ticker)
            continue

        sig = get_latest_signal(daily_df)

        # ② 매도 신호 체크 (보유 종목만)
        if ticker in positions:
            if sig["sell_full"]:
                logger.info("  전략 매도(1원칙) 실행: %s", ticker)
                execute_sell_all(ticker)
                msg = build_sell_full_message(stock_name, sig, "전략 매도 신호(1원칙)")
                send_telegram(msg)
                clear_position(ticker)
                continue
            if sig["sell_partial"]:
                logger.info("  전략 매도(2원칙) 실행: %s", ticker)
                execute_sell_all(ticker)
                msg = build_sell_partial_message(stock_name, sig, "전략 매도 신호(2원칙)")
                send_telegram(msg)
                clear_position(ticker)
                continue

        # ③ 매수 신호 체크 (미보유 종목만)
        if ticker not in positions:
            buy_triggered = False
            buy_principle = ""

            try:
                if minute_df is not None and not minute_df.empty:
                    if buy_signal_1_intraday(ticker, minute_df, daily_df):
                        buy_triggered = True
                        buy_principle = "1원칙(분봉)"
            except Exception as e:
                logger.warning("  분봉 매수 신호 확인 실패: %s", e)

            if not buy_triggered:
                last = daily_df.iloc[-1]
                if last.get("buy_signal_2", False):
                    buy_triggered = True
                    buy_principle = "2원칙(눌림목)"
                elif last.get("buy_signal_3", False):
                    buy_triggered = True
                    buy_principle = "3원칙(거래량급증)"

            if buy_triggered:
                qty = calc_position_size(ticker, available_cash, current_price)
                if qty > 0:
                    result = execute_buy(stock_code, qty)
                    if result:
                        register_position(ticker, current_price, qty)
                        msg = build_buy_message(stock_name, sig, buy_principle)
                        send_telegram(msg)
                        logger.info("  매수 실행: %s %d주 @ %,.0f원 (%s)",
                                    stock_code, qty, current_price, buy_principle)
                else:
                    logger.info("  매수 신호(%s) — 자금 부족 또는 이미 보유", buy_principle)

        logger.info("  완료: 현재가=%,.0f원", current_price)

    logger.info("\n매매 루프 종료")


def main():
    logger.info("스케줄러 시작 — 30분 간격 장중 실행")

    times = []
    h, m = 9, 5
    while (h, m) <= (15, 5):
        times.append(f"{h:02d}:{m:02d}")
        m += 30
        if m >= 60:
            m -= 60
            h += 1

    schedule.clear()
    for t in times:
        schedule.every().day.at(t).do(run_priority_loop)
        logger.info("  등록: %s", t)

    schedule.every().day.at("15:00").do(send_daily_summary)
    logger.info("  등록: 15:00 (일일 기술적 분석 리포트)")

    logger.info("총 %d개 시간대 등록 완료", len(times))

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
