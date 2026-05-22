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
from morning_briefer import send_morning_briefing
from trade_logger import log_buy, log_sell
from conditional_orders import check_and_execute as check_cond_orders
from signals.scanner import scan_all
from signals.alert import send_signal_alert
from config import GROWTH_ASSET_RATIO

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
    """분봉 거래량 급증(5배↑) 감지 → 네이버 뉴스 3건 텔레그램 전송."""
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
                log_sell(ticker, sig["close"], notes="2원칙-전량")
                clear_position(ticker)
                continue
            if sig["sell_partial"]:
                logger.info("  전략 매도(1원칙-부분) 실행: %s", ticker)
                execute_sell_half(ticker)
                msg = build_sell_partial_message(stock_name, sig, "전략 매도 신호(1원칙-부분)")
                send_telegram(msg)
                log_sell(ticker, sig["close"], notes="1원칙-부분")
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
                        log_buy(ticker, stock_name, current_price, qty, strategy=buy_principle)
                        logger.info("  매수 실행: %s %d주 @ %s원 (%s)",
                                    stock_code, qty, f"{current_price:,.0f}", buy_principle)
                else:
                    logger.info("  매수 신호(%s) — 자금 부족 또는 이미 보유", buy_principle)

    logger.info("\n매매 루프 종료")


# ─────────────────────────────────────────────
# 봇 활성화 게이트
# ─────────────────────────────────────────────

import json as _json

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
        return  # 이미 초기화됨

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

    legacy = state.get("legacy_tickers", [])
    if not legacy:
        # legacy_tickers 없음 = 처음부터 빈 계좌 → 바로 활성화
        state["bot_active"] = True
        state["activated_at"] = datetime.now(KST).isoformat()
        _save_state(state)
        return True

    if not KIS_APP_KEY:
        return False

    try:
        balance  = KISTrader().get_balance()
        held     = {h["stock_code"] for h in balance if h.get("qty", 0) > 0}
        still_holding = [t for t in legacy if t in held]

        if not still_holding:
            state["bot_active"]   = True
            state["activated_at"] = datetime.now(KST).isoformat()
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
# 장 시간 (한국 / 미국)
# ─────────────────────────────────────────────

def _is_kr_market() -> bool:
    now = datetime.now(KST)
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    return 9 * 60 <= t <= 15 * 60 + 30


def _is_us_market() -> bool:
    """미국 장 여부 (서머타임 자동 감지)."""
    now = datetime.now(KST)
    if now.weekday() >= 5:
        return False
    t = now.hour * 60 + now.minute
    # 서머타임(3~11월): 22:30~05:00 / 동절기: 23:30~06:00
    import pytz
    eastern = pytz.timezone("America/New_York")
    now_et  = now.astimezone(eastern)
    dst     = bool(now_et.dst())
    if dst:
        return t >= 22 * 60 + 30 or t <= 5 * 60
    else:
        return t >= 23 * 60 + 30 or t <= 6 * 60


# ─────────────────────────────────────────────
# 급등주 10분 스캔
# ─────────────────────────────────────────────

def scan_growth_signals():
    """
    10분마다 실행 — 한국장 또는 미국장 시간 중에만 스캔.
    봇이 활성화된 경우에만 동작.
    """
    if not _check_activation():
        return

    if not (_is_kr_market() or _is_us_market()):
        return

    now = datetime.now(KST)
    logger.info("급등주 신호 스캔 (%s)", now.strftime("%H:%M"))

    growth_cash = 10_000_000 * GROWTH_ASSET_RATIO
    if KIS_APP_KEY:
        try:
            t           = KISTrader()
            total_asset = _get_total_asset(t)
            growth_cash = total_asset * GROWTH_ASSET_RATIO
        except Exception as e:
            logger.warning("총 자산 조회 실패: %s", e)

    from data_fetcher import fetch_ohlcv as _fetch
    signals = scan_all(STOCKS, lambda tk: _fetch(tk, period_years=1))

    if not signals:
        logger.debug("급등주 신호 없음")
        return

    logger.info("신호 발생: %d개 종목", len(signals))
    for sig in signals:
        send_signal_alert(sig, growth_cash)


# ─────────────────────────────────────────────
# 월간 리밸런싱
# ─────────────────────────────────────────────

def run_monthly_rebalance():
    """매월 1일 오전 8시 30분 실행 — 안전자산 비중 재조정."""
    if not is_bot_active():
        return

    now = datetime.now(KST)
    if now.weekday() >= 5:
        return

    logger.info("월간 리밸런싱 시작")
    try:
        from portfolio.rebalancer import run_monthly_rebalance as _rebalance

        holdings = {}
        total_asset = 10_000_000
        if KIS_APP_KEY:
            t        = KISTrader()
            balance  = t.get_balance()
            cash     = t.get_available_cash()
            kr_val   = sum(h["qty"] * h["avg_price"] for h in balance)
            holdings = {
                h["stock_code"]: {"qty": h["qty"], "avg_price": h["avg_price"]}
                for h in balance
            }

            # 미국주식 잔고 포함 (통합증거금)
            try:
                us_balance = t.get_us_balance()
                for h in us_balance:
                    sym = h["symbol"]
                    # USD 평가액을 원화로 환산 (근사치 1400원/달러)
                    usd_val = h["qty"] * h["avg_price"]
                    holdings[sym] = {"qty": h["qty"], "avg_price": h["avg_price"], "currency": "USD"}
                    kr_val += usd_val * 1400
            except Exception as e:
                logger.warning("미국주식 잔고 조회 실패 (리밸런싱 제외): %s", e)

            total_asset = cash + kr_val

        _rebalance(total_asset, holdings)
    except Exception as e:
        logger.error("월간 리밸런싱 실패: %s", e)
        send_telegram(f"⚠️ 월간 리밸런싱 오류: {e}")


def main():
    logger.info("스케줄러 시작")

    # 기존 보유 종목 기록 (최초 1회)
    _init_legacy_tickers()

    # 한국장 5분 간격 매매 루프 (09:05 ~ 15:25)
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

    # 고정 스케줄
    schedule.every().day.at("08:00").do(send_morning_briefing)
    schedule.every().day.at("15:00").do(send_daily_summary)

    # 급등주 10분 스캔 (장중 자동 필터링)
    schedule.every(10).minutes.do(scan_growth_signals)

    # 월간 리밸런싱 (매월 1일 08:30)
    schedule.every().day.at("08:30").do(
        lambda: run_monthly_rebalance() if datetime.now(KST).day == 1 else None
    )

    logger.info(
        "등록 완료: 매매루프 %d개 / 08:00 모닝브리핑 / 15:00 일일리포트 / "
        "10분 급등주스캔 / 매월 1일 08:30 리밸런싱",
        len(times),
    )

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
