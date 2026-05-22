"""
runner.py - ML 급등주 전략 스케줄러

스케줄:
  5분마다  → scan_growth_signals() : ML 급등주 스캔 (한국장·미국장 자동 필터)
  08:00   → send_morning_briefing()
  08:30   → run_monthly_rebalance() (매월 1일)
  15:00   → send_daily_summary()
"""

import schedule
import time
import logging
import logging.handlers
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date

import pandas as pd
import pytz
import yfinance as yf

from config import (
    STOCKS, US_STOCKS, MA_SHORT, MA_LONG, RSI_PERIOD, KIS_APP_KEY, GROWTH_ASSET_RATIO,
)
from data_fetcher import fetch_ohlcv, get_minute_data
from indicators import add_all_indicators, detect_crossover
from strategy import generate_signals, get_latest_signal
from trader import positions, KISTrader
from notifier import send_telegram, build_daily_summary_message
from morning_briefer import send_morning_briefing
from signals.scanner import scan_all
from signals.alert import send_signal_alert

KST      = pytz.timezone("Asia/Seoul")
LOG_FILE = "logs/trader.log"

# OHLCV 캐시: {ticker: (DataFrame, timestamp)}  — 30분 TTL
_OHLCV_CACHE: dict = {}
_CACHE_TTL   = 1800  # 초

# 분봉 캐시: {ticker: (DataFrame, timestamp)}  — 5분 TTL
_MINUTE_CACHE: dict = {}
_MINUTE_TTL   = 300  # 초


def _fetch_minute_yf(ticker: str):
    """yfinance 5분봉 다운로드 (한국/미국 공통, ~15분 지연). 캐시 5분."""
    now = time.time()
    if ticker in _MINUTE_CACHE:
        df, ts = _MINUTE_CACHE[ticker]
        if now - ts < _MINUTE_TTL:
            return df
    try:
        df = yf.download(ticker, period="1d", interval="5m",
                         auto_adjust=True, progress=False)
        if df.empty:
            _MINUTE_CACHE[ticker] = (None, now)
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={
            "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        _MINUTE_CACHE[ticker] = (df, now)
        return df
    except Exception:
        _MINUTE_CACHE[ticker] = (None, now)
        return None


def _cached_fetch(ticker: str, period_years: int = 1) -> pd.DataFrame:
    now = time.time()
    if ticker in _OHLCV_CACHE:
        df, ts = _OHLCV_CACHE[ticker]
        if now - ts < _CACHE_TTL:
            return df
    df = fetch_ohlcv(ticker, period_years)
    _OHLCV_CACHE[ticker] = (df, time.time())
    return df


def _prefetch_parallel(tickers: list, period_years: int = 1, max_workers: int = 15):
    """캐시 미스(만료·신규) 종목만 병렬 다운로드 후 캐시에 저장."""
    now   = time.time()
    stale = [t for t in tickers
             if t not in _OHLCV_CACHE or now - _OHLCV_CACHE[t][1] >= _CACHE_TTL]
    if not stale:
        return

    logger.info("OHLCV 병렬 다운로드: %d종목 (workers=%d)", len(stale), max_workers)

    def _fetch_one(ticker):
        df = fetch_ohlcv(ticker, period_years)
        _OHLCV_CACHE[ticker] = (df, time.time())

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in stale}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.debug("prefetch 실패 [%s]: %s", futures[fut], e)

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
    5분마다 실행 — 한국장 또는 미국장 시간 중에만 스캔.
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

    if _is_kr_market():
        # 한국장: KRX 전체 종목 1차 스크리닝 후 ML 분석
        from signals.krx_universe import get_krx_candidates
        stocks_to_scan = get_krx_candidates(top_n=100)
        if not stocks_to_scan:
            stocks_to_scan = STOCKS  # fallback
    else:
        # 미국장: S&P 500 전체 스크리닝
        from signals.us_universe import get_us_candidates
        stocks_to_scan = get_us_candidates(top_n=50)
        if not stocks_to_scan:
            stocks_to_scan = US_STOCKS if US_STOCKS else STOCKS

    # 일봉 캐시 미스 종목 병렬 다운로드
    tickers = list(stocks_to_scan.keys())
    _prefetch_parallel(tickers)

    # 분봉 병렬 다운로드 (오늘 바 합성용, 5분 TTL)
    now = time.time()
    stale_min = [t for t in tickers
                 if t not in _MINUTE_CACHE or now - _MINUTE_CACHE[t][1] >= _MINUTE_TTL]
    if stale_min:
        logger.info("분봉 병렬 다운로드: %d종목", len(stale_min))
        with ThreadPoolExecutor(max_workers=10) as ex:
            list(ex.map(_fetch_minute_yf, stale_min))

    def _fetch_with_today(ticker: str) -> pd.DataFrame:
        df = _cached_fetch(ticker)
        minute_df = _fetch_minute_yf(ticker)
        return _append_today_bar(df, minute_df)

    signals = scan_all(stocks_to_scan, _fetch_with_today)

    if not signals:
        logger.debug("급등주 신호 없음")
        return

    logger.info("신호 발생: %d개 종목", len(signals))
    for sig in signals:
        send_signal_alert(sig, growth_cash)


# ─────────────────────────────────────────────
# 월간 리밸런싱
# ─────────────────────────────────────────────

def retrain_models():
    """매일 07:30 실행 — 전체 종목 ML 모델 최신 데이터로 재학습."""
    now = datetime.now(KST)
    if now.weekday() >= 5:
        return
    logger.info("ML 일일 재학습 시작")
    try:
        from ml.trainer import retrain_daily
        results = retrain_daily()
        ok   = sum(1 for v in results.values() if v)
        fail = len(results) - ok
        send_telegram(
            f"🤖 <b>ML 모델 재학습 완료</b>\n"
            f"성공: {ok}개 / 실패: {fail}개\n"
            f"{now.strftime('%Y-%m-%d %H:%M')} 기준 최신 데이터 반영"
        )
    except Exception as e:
        logger.error("ML 재학습 실패: %s", e)
        send_telegram(f"⚠️ ML 재학습 오류: {e}")


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

    schedule.clear()

    # 고정 스케줄
    schedule.every().day.at("07:30").do(retrain_models)
    schedule.every().day.at("08:00").do(send_morning_briefing)
    schedule.every().day.at("15:00").do(send_daily_summary)

    # ML 급등주 5분 스캔 (한국장 + 미국장 자동 필터링)
    schedule.every(5).minutes.do(scan_growth_signals)

    # 월간 리밸런싱 (매월 1일 08:30)
    schedule.every().day.at("08:30").do(
        lambda: run_monthly_rebalance() if datetime.now(KST).day == 1 else None
    )

    logger.info(
        "등록 완료: 07:30 ML재학습 / 08:00 모닝브리핑 / 15:00 일일리포트 / "
        "5분 ML 급등주스캔 / 매월 1일 08:30 리밸런싱"
    )

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
