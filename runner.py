"""
runner.py - 장중 우선순위 기반 자동매매 스케줄러
실행 순서 (매 사이클):
  ① 매도신호 → ② 매수신호

분봉 데이터로 오늘 실시간 일봉 바를 구성해 모든 신호를 현재 시각 기준으로 판단.
분봉 거래량 급증 감지 시 네이버 최신 뉴스 3건을 텔레그램으로 전송.
"""

import schedule
import time
import logging
import logging.handlers
from datetime import datetime, timedelta, date

import pandas as pd
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
    calc_position_size,
    register_position, clear_position,
    execute_buy, execute_sell_all, execute_sell_half,
)
from notifier import (
    build_buy_message, build_sell_full_message, build_sell_partial_message,
    send_telegram, build_volume_surge_message, build_daily_summary_message,
)
from news_fetcher import fetch_naver_news
from conditional_orders import check_and_execute as check_cond_orders

KST      = pytz.timezone("Asia/Seoul")
LOG_FILE = "logs/trader.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        ),
    ],
    force=True,
)
logger = logging.getLogger("runner")

_surge_alerted: dict = {}
_SURGE_COOLDOWN_MINUTES = 30


def is_market_hours() -> bool:
    now = datetime.now(KST)
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def _get_total_asset(trader: KISTrader) -> float:
    try:
        cash      = trader.get_available_cash()
        balance   = trader.get_balance()
        stock_val = sum(b["qty"] * b["avg_price"] for b in balance)
        return float(cash + stock_val)
    except Exception as e:
        logger.warning("총 자산 조회 실패: %s", e)
        return 0.0


def _append_today_bar(daily_df: pd.DataFrame, minute_df) -> pd.DataFrame:
    """
    분봉 데이터로 오늘 실시간 일봉 바를 구성해 daily_df 마지막 행에 반영.
    - 오늘 날짜 행이 이미 있으면 업데이트, 없으면 새 행 추가.
    - 지표 계산 전에 호출해야 함.
    """
    if minute_df is None or minute_df.empty:
        return daily_df

    today = date.today()

    try:
        today_open   = float(minute_df["open"].iloc[0])
        today_high   = float(minute_df["high"].max())
        today_low    = float(minute_df["low"].min())
        today_close  = float(minute_df["close"].iloc[-1])
        today_volume = int(minute_df["volume"].sum())
    except Exception:
        return daily_df

    daily_df = daily_df.copy()

    if daily_df.index[-1].date() == today:
        # 이미 오늘 행이 있으면 실시간 값으로 덮어씀
        idx = daily_df.index[-1]
        daily_df.loc[idx, "Open"]   = today_open
        daily_df.loc[idx, "High"]   = today_high
        daily_df.loc[idx, "Low"]    = today_low
        daily_df.loc[idx, "Close"]  = today_close
        daily_df.loc[idx, "Volume"] = today_volume
    else:
        # 오늘 행이 없으면 신규 추가
        new_idx = pd.Timestamp(today)
        new_row = pd.DataFrame(
            [[today_open, today_high, today_low, today_close, today_volume]],
            columns=["Open", "High", "Low", "Close", "Volume"],
            index=[new_idx],
        )
        # 나머지 컬럼은 NaN으로 채워짐 (지표 계산 시 덮어씌워짐)
        daily_df = pd.concat([daily_df, new_row])

    return daily_df


def _check_volume_surge(ticker: str, stock_name: str, minute_df, current_price: float):
    """분봉 거래량 급증(3.5배↑) 감지 → 네이버 뉴스 3건 텔레그램 전송."""
    if minute_df is None or minute_df.empty or len(minute_df) < 6:
        return

    current_vol = float(minute_df["volume"].iloc[-1])
    avg_vol     = float(minute_df["volume"].iloc[-6:-1].mean())

    if avg_vol <= 0:
        return

    surge_ratio = current_vol / avg_vol
    if surge_ratio < VOLUME_SURGE_MINUTE_RATIO:
        return

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

            # 오늘 실시간 바 반영 (분봉 기반)
            minute_df = None
            try:
                minute_df = get_minute_data(ticker, interval_min=1)
            except Exception:
                pass
            daily_df = _append_today_bar(daily_df, minute_df)

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
            trader      = KISTrader()
            total_asset = _get_total_asset(trader)
        except Exception as e:
            logger.error("KISTrader 초기화 실패: %s", e)
            return
    else:
        trader      = None
        total_asset = 10_000_000
        logger.info("KIS_APP_KEY 미설정 — 시뮬레이션 모드")

    for ticker, stock_name in STOCKS.items():
        stock_code = ticker.replace(".KS", "").replace(".KQ", "")
        logger.info("\n>>> [%s] %s 처리 시작", ticker, stock_name)

        # ── 1단계: 원시 일봉 데이터 조회 ─────────────────
        try:
            raw_df = fetch_ohlcv(ticker, period_years=1)
        except Exception as e:
            logger.error("  일봉 데이터 조회 실패: %s", e)
            continue

        # ── 2단계: 현재가 + 분봉 데이터 조회 ─────────────
        minute_df     = None
        current_price = float(raw_df["Close"].iloc[-1])
        try:
            if trader:
                price_info    = trader.get_current_price(stock_code)
                current_price = float(price_info["price"])
            try:
                minute_df = get_minute_data(ticker, interval_min=1)
            except Exception as e:
                logger.warning("  분봉 데이터 조회 실패: %s", e)
        except Exception as e:
            logger.error("  현재가 조회 실패: %s", e)
            continue

        # ── 3단계: 오늘 실시간 바 추가 → 지표 재계산 ─────
        daily_df = _append_today_bar(raw_df, minute_df)
        try:
            daily_df = add_all_indicators(
                daily_df, short=MA_SHORT, long=MA_LONG, rsi_period=RSI_PERIOD
            )
            daily_df = detect_crossover(daily_df, short=MA_SHORT, long=MA_LONG)
            daily_df = generate_signals(daily_df)
        except Exception as e:
            logger.error("  지표/신호 계산 실패: %s", e)
            continue

        # ── 분봉 거래량 급증 감지 ────────────────────────
        if minute_df is not None and not minute_df.empty:
            _check_volume_surge(ticker, stock_name, minute_df, current_price)

        # ── 조건부 주문 체크 ──────────────────────────
        for cond_msg in check_cond_orders(ticker, stock_code, current_price, trader):
            send_telegram(cond_msg)

        sig = get_latest_signal(daily_df)
        logger.info("  실시간 기준일: %s  현재가: %s원", sig["date"], f"{current_price:,.0f}")

        # ① 매도 신호 체크 (보유 종목만)
        if ticker in positions:
            if sig["sell_full"]:
                logger.info("  전략 매도(2원칙-전량) 실행: %s", ticker)
                execute_sell_all(ticker)
                msg = build_sell_full_message(stock_name, sig, "전략 매도 신호(2원칙-전량)")
                send_telegram(msg)
                clear_position(ticker)
                continue
            if sig["sell_partial"]:
                logger.info("  전략 매도(1원칙-부분) 실행: %s", ticker)
                execute_sell_half(ticker)
                msg = build_sell_partial_message(stock_name, sig, "전략 매도 신호(1원칙-부분)")
                send_telegram(msg)
                continue

        # ② 매수 신호 체크 (미보유 종목만)
        if ticker not in positions:
            buy_triggered = False
            buy_principle = ""

            # 1원칙: 분봉 기반 (9:30 이후 시가 돌파)
            try:
                if minute_df is not None and not minute_df.empty:
                    if buy_signal_1_intraday(ticker, minute_df, daily_df):
                        buy_triggered = True
                        buy_principle = "1원칙(시가돌파)"
            except Exception as e:
                logger.warning("  분봉 매수 신호 확인 실패: %s", e)

            # 2, 3원칙: 실시간 일봉 기반
            if not buy_triggered:
                last = daily_df.iloc[-1]
                if last.get("buy_signal_2", False):
                    buy_triggered = True
                    buy_principle = "2원칙(MA사이반등)"
                elif last.get("buy_signal_3", False):
                    buy_triggered = True
                    buy_principle = "3원칙(MA20아래급등)"

            if buy_triggered:
                qty = calc_position_size(ticker, total_asset, current_price)
                if qty > 0:
                    result = execute_buy(stock_code, qty)
                    if result:
                        register_position(ticker, current_price, qty)
                        msg = build_buy_message(stock_name, sig, buy_principle)
                        send_telegram(msg)
                        logger.info("  매수 실행: %s %d주 @ %s원 (%s)",
                                    stock_code, qty, f"{current_price:,.0f}", buy_principle)
                else:
                    logger.info("  매수 신호(%s) — 자금 부족 또는 이미 보유", buy_principle)

    logger.info("\n매매 루프 종료")


def main():
    logger.info("스케줄러 시작 — 5분 간격 장중 실행")

    times = []
    h, m = 9, 5
    while (h, m) <= (15, 25):
        times.append(f"{h:02d}:{m:02d}")
        m += 5
        if m >= 60:
            m -= 60
            h += 1

    schedule.clear()
    for t in times:
        schedule.every().day.at(t).do(run_priority_loop)

    schedule.every().day.at("15:00").do(send_daily_summary)

    logger.info("총 %d개 시간대 등록 완료 (+ 15:00 일일 리포트)", len(times))

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
