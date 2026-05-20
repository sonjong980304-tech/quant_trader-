"""
runner.py - 장중 우선순위 기반 자동매매 스케줄러
실행 순서 (매 사이클):
  ① 고점 갱신 → ② 손절 경고 → ③ 손절 → ④ 익절 → ⑤ 매도신호 → ⑥ 매수신호

손절/익절은 LangGraph/AI 판단 없이 즉시 실행.
매수/매도 신호는 전략 함수 기반으로 직접 실행.
"""

import schedule
import time
import logging
import logging.handlers
from datetime import datetime

import pytz

from config import (
    STOCKS, MA_SHORT, MA_LONG, RSI_PERIOD, KIS_APP_KEY,
)
from data_fetcher import fetch_ohlcv, get_minute_data
from indicators import add_all_indicators, detect_crossover
from strategy import generate_signals, get_latest_signal, buy_signal_1_intraday
from trader import (
    positions, KISTrader,
    update_highest_price, check_stop_loss, check_stop_loss_warning,
    check_take_profit, calc_position_size,
    register_position, clear_position,
    execute_buy, execute_sell_all, execute_sell_half,
)
from notifier import (
    send_stop_loss_alert, send_stop_loss_warning, send_take_profit_alert,
    build_buy_message, build_sell_full_message, build_sell_partial_message,
    send_telegram,
)

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


def is_market_hours() -> bool:
    """현재 시각이 한국 주식 장 시간(09:00~15:30, 평일)인지 확인"""
    now = datetime.now(KST)
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def _get_total_asset(trader: KISTrader) -> float:
    """주문 가능 현금 + 보유 종목 평가금액 합산"""
    try:
        cash    = trader.get_available_cash()
        balance = trader.get_balance()
        stock_val = sum(
            b["qty"] * b["avg_price"] for b in balance
        )
        return float(cash + stock_val)
    except Exception as e:
        logger.warning("총 자산 조회 실패: %s", e)
        return 0.0


def run_priority_loop():
    """
    우선순위 기반 매매 루프.
    장 시간 외에는 자동으로 건너뜀.
    """
    if not is_market_hours():
        logger.info("장 시간 외 — 실행 건너뜀 (%s)",
                    datetime.now(KST).strftime("%H:%M"))
        return

    now = datetime.now(KST)
    logger.info("=" * 55)
    logger.info("매매 루프 시작 (%s)", now.strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 55)

    # KIS API 초기화
    if KIS_APP_KEY:
        try:
            trader          = KISTrader()
            available_cash  = trader.get_available_cash()
        except Exception as e:
            logger.error("KISTrader 초기화 실패: %s", e)
            return
    else:
        trader         = None
        available_cash = 10_000_000  # 시뮬레이션용 기본 가용 현금
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

        # ── 현재가 / MA5 조회 ─────────────────────────────
        try:
            if trader:
                price_info    = trader.get_current_price(stock_code)
                current_price = float(price_info["price"])
            else:
                current_price = float(daily_df["Close"].iloc[-1])
            ma5 = float(daily_df[f"MA_{MA_SHORT}"].iloc[-1])
        except Exception as e:
            logger.error("  현재가 조회 실패: %s", e)
            continue

        # ① 고점 갱신 (매 루프 최우선)
        update_highest_price(ticker, current_price)

        # ② 손절 경고 (주문 없이 알림만)
        if check_stop_loss_warning(ticker, current_price):
            avg_price  = positions[ticker]["avg_price"]
            loss_ratio = (current_price - avg_price) / avg_price
            send_stop_loss_warning(stock_name, current_price, loss_ratio)
            logger.warning("  손절 경고: %.1f%%", loss_ratio * 100)

        # ③ 손절 체크 (즉시 실행, LangGraph 우회)
        stop_type, reason = check_stop_loss(ticker, current_price, ma5)
        if stop_type:
            logger.warning("  손절 실행: %s | %s", stop_type, reason)
            execute_sell_all(ticker)
            send_stop_loss_alert(stock_name, current_price, stop_type, reason)
            clear_position(ticker)
            continue

        # ④ 익절 체크 (즉시 실행)
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

        # ⑤ 매도 신호 체크 (보유 종목만)
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

        # ⑥ 매수 신호 체크 (미보유 종목만)
        if ticker not in positions:
            buy_triggered = False
            buy_principle = ""

            # 1원칙: 분봉 기반 (장 시작 직후 패턴)
            try:
                minute_df = get_minute_data(ticker, interval_min=1)
                if not minute_df.empty:
                    if buy_signal_1_intraday(ticker, minute_df, daily_df):
                        buy_triggered = True
                        buy_principle = "1원칙(분봉)"
            except Exception as e:
                logger.warning("  분봉 데이터 조회 실패: %s", e)

            # 2, 3원칙: 일봉 기반
            if not buy_triggered:
                last = daily_df.iloc[-1]
                if last.get("buy_signal_2", False):
                    buy_triggered = True
                    buy_principle = "2원칙(눌림목)"
                elif last.get("buy_signal_3", False):
                    buy_triggered = True
                    buy_principle = "3원칙(급락반등)"

            if buy_triggered:
                qty = calc_position_size(ticker, available_cash, current_price)
                if qty > 0:
                    result = execute_buy(stock_code, qty)
                    if result:
                        register_position(ticker, current_price, qty)
                        msg = build_buy_message(
                            stock_name, sig, "중립적",
                            f"전략 매수 신호 ({buy_principle})"
                        )
                        send_telegram(msg)
                        logger.info("  매수 실행: %s %d주 @ %,.0f원 (%s)",
                                    stock_code, qty, current_price, buy_principle)
                else:
                    logger.info("  매수 신호(%s) — 자금 부족 또는 이미 보유", buy_principle)

        logger.info("  완료: 신호=%s%s, 현재가=%,.0f원",
                    sig["signal_type"] if "signal_type" in sig else
                    ("매수" if sig["buy"] else "매도" if sig["sell_full"] or sig["sell_partial"] else "없음"),
                    f"({'|'.join(sig.get('buy_which') or sig.get('sell_which') or [])})" if
                    (sig.get('buy_which') or sig.get('sell_which')) else "",
                    current_price)

    logger.info("\n매매 루프 종료")


def main():
    logger.info("스케줄러 시작 — 30분 간격 장중 실행 (우선순위 기반)")

    # 30분마다 실행: 09:05, 09:35, ..., 15:05
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

    logger.info("총 %d개 시간대 등록 완료", len(times))

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
