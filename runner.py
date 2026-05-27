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
from datetime import datetime, date, timedelta

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
from market_calendar import is_kr_trading_day

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
        # yfinance 1.x 간헐적 중복 타임스탬프 제거
        df = df[~df.index.duplicated(keep="last")]
        df = df.rename(columns={
            "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        _MINUTE_CACHE[ticker] = (df, now)
        return df
    except Exception:
        _MINUTE_CACHE[ticker] = (None, now)
        return None


def _fetch_kr_realtime_bar(ticker: str):
    """
    KIS API로 한국 주식 당일 실시간 바 생성 (yfinance 실패 시 fallback).
    반환: 단일 행 DataFrame (open/high/low/close/volume) 또는 None
    """
    if not KIS_APP_KEY:
        return None
    try:
        code = ticker.replace(".KS", "").replace(".KQ", "")
        info = KISTrader().get_current_price(code)
        row = pd.DataFrame([{
            "open":   info["open"]  or info["price"],
            "high":   info["high"]  or info["price"],
            "low":    info["low"]   or info["price"],
            "close":  info["price"],
            "volume": info["volume"],
        }])
        return row
    except Exception as e:
        logger.debug("KIS 실시간 바 조회 실패 [%s]: %s", ticker, e)
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


def _prefetch_parallel(tickers: list, period_years: int = 1, max_workers: int = 5):
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
    """총평가금액(예수금+주식평가액) 조회.
    tot_evlu_amt가 0이면(미체결 등으로 주식 미보유) 예수금으로 폴백."""
    try:
        total = float(trader.get_total_eval_amt())
        if total > 0:
            return total
        # tot_evlu_amt=0 → 보유 주식 없음, 예수금만으로 계산
        return float(trader.get_available_cash())
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
    # 컬럼 중복 방어 (yfinance 간헐적 오염)
    if daily_df.columns.duplicated().any():
        daily_df = daily_df.loc[:, ~daily_df.columns.duplicated(keep="last")]

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

    # concat 후 중복 인덱스 제거 (yfinance가 당일 바를 이미 포함한 경우 대비)
    daily_df = daily_df[~daily_df.index.duplicated(keep="last")]
    return daily_df


def send_daily_summary():
    """오후 3시 현재 보유 종목 기준 일일 기술적 분석 리포트를 텔레그램으로 전송."""
    now = datetime.now(KST)
    if not is_kr_trading_day(now.date()):
        return

    ml_positions = _load_state().get("ml_positions", {})
    if not ml_positions:
        send_telegram(
            f"📊 <b>일일 기술적 분석 리포트</b>\n"
            f"{now.strftime('%Y-%m-%d %H:%M')} 기준\n"
            f"현재 보유 종목 없음"
        )
        return

    logger.info("일일 기술적 분석 리포트 전송 시작")
    send_telegram(
        f"📊 <b>일일 기술적 분석 리포트</b> (보유 {len(ml_positions)}종목)\n"
        f"{now.strftime('%Y-%m-%d %H:%M')} 기준"
    )

    for ticker, pos_info in ml_positions.items():
        stock_name = pos_info.get("name", ticker)
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
                stock_name, sig, daily_df, position=pos_info
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
# ML 포지션 추적 (익절 / 7거래일 자동 청산)
# ─────────────────────────────────────────────

def save_ml_position(ticker: str, name: str, qty: int, entry_price: float,
                     avg_win: float, is_us: bool = False):
    """매수 확정 시 ML 포지션 저장."""
    state = _load_state()
    positions = state.setdefault("ml_positions", {})
    target_price = entry_price * (1 + avg_win)
    stop_price   = entry_price * 0.93  # 손절 -7%
    positions[ticker] = {
        "ticker":       ticker,
        "name":         name,
        "qty":          qty,
        "entry_price":  round(entry_price, 4),
        "target_price": round(target_price, 4),
        "stop_price":   round(stop_price, 4),
        "entry_date":   datetime.now(KST).strftime("%Y-%m-%d"),
        "is_us":        is_us,
    }
    _save_state(state)
    logger.info("ML 포지션 저장: %s qty=%d entry=%.2f target=%.2f stop=%.2f",
                ticker, qty, entry_price, target_price, stop_price)


def _get_current_price(ticker: str):
    """현재가 조회 (한국 주식: KIS API, 미국 주식: yfinance)."""
    if KIS_APP_KEY and (ticker.endswith(".KS") or ticker.endswith(".KQ")):
        try:
            code = ticker.replace(".KS", "").replace(".KQ", "")
            return float(KISTrader().get_current_price(code)["price"])
        except Exception:
            pass
    try:
        info = yf.Ticker(ticker).fast_info
        return float(info["lastPrice"])
    except Exception:
        return None


def _trading_days_elapsed(entry_date_str: str) -> int:
    """입력 날짜부터 오늘까지 거래일(평일) 수."""
    from datetime import date as date_cls
    start = date_cls.fromisoformat(entry_date_str)
    today = date_cls.today()
    days  = 0
    d     = start
    while d < today:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days += 1
    return days


def check_ml_positions():
    """
    5분마다 실행 — ML 포지션 익절 / 손절 / 7거래일 자동 청산 체크.
    - 현재가 >= target_price  → 익절 매도
    - 현재가 <= stop_price    → 손절 매도 (-7%)
    - 7거래일 경과             → 강제 청산
    """
    state     = _load_state()
    positions = state.get("ml_positions", {})
    if not positions:
        return

    to_remove = []
    for ticker, pos in positions.items():
        try:
            cur_price  = _get_current_price(ticker)
            elapsed    = _trading_days_elapsed(pos["entry_date"])
            qty        = pos["qty"]
            name       = pos.get("name", ticker)
            target     = pos["target_price"]
            stop       = pos.get("stop_price", pos["entry_price"] * 0.93)
            is_us      = pos.get("is_us", False)

            reason = None
            emoji  = "⏰"
            if cur_price and cur_price >= target:
                reason = f"익절 (현재가 {cur_price:.2f} ≥ 목표 {target:.2f})"
                emoji  = "✅"
            elif cur_price and cur_price <= stop:
                reason = f"손절 (현재가 {cur_price:.2f} ≤ 손절가 {stop:.2f})"
                emoji  = "🔴"
            elif elapsed >= 7:
                reason = f"7거래일 경과 ({elapsed}일)"

            if reason is None:
                continue

            # 매도 실행
            logger.info("[%s] 자동 매도 — %s", ticker, reason)
            sell_price = cur_price or pos["entry_price"]
            if KIS_APP_KEY:
                t = KISTrader()
                code = ticker.replace(".KS", "").replace(".KQ", "")
                if is_us:
                    t.sell_us(code, qty)
                else:
                    t.sell(code, qty)

            # CSV 매도 기록
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
            # 실제 보유 없음 → 포지션 동기화 (KIS "수량 초과" = 실제 미보유)
            if "수량" in str(e) and "초과" in str(e):
                logger.warning("[%s] 실제 미보유 확인 — state.json에서 제거", ticker)
                to_remove.append(ticker)

    if to_remove:
        for t in to_remove:
            positions.pop(t, None)
        state["ml_positions"] = positions
        _save_state(state)


# ─────────────────────────────────────────────
# 장 시간 (한국 / 미국)
# ─────────────────────────────────────────────

def _is_kr_market() -> bool:
    now = datetime.now(KST)
    if now.weekday() >= 5:
        return False
    if not is_kr_trading_day(now.date()):
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

    # 이미 보유 중인 ML 포지션 종목 스캔에서 제외 (당일 중복 매수 방지)
    held = set(_load_state().get("ml_positions", {}).keys())
    if held:
        before = len(stocks_to_scan)
        stocks_to_scan = {t: n for t, n in stocks_to_scan.items() if t not in held}
        logger.info("보유 중 종목 제외: %d개 → %d개", before, len(stocks_to_scan))

    # 일봉 캐시 미스 종목 병렬 다운로드
    tickers = list(stocks_to_scan.keys())
    _prefetch_parallel(tickers)

    # 미국 주식만 yfinance 분봉 병렬 다운로드 (한국 주식은 KIS API 실시간 사용)
    us_tickers = [t for t in tickers if not (t.endswith(".KS") or t.endswith(".KQ"))]
    now = time.time()
    stale_min = [t for t in us_tickers
                 if t not in _MINUTE_CACHE or now - _MINUTE_CACHE[t][1] >= _MINUTE_TTL]
    if stale_min:
        logger.info("미국 분봉 병렬 다운로드: %d종목", len(stale_min))
        with ThreadPoolExecutor(max_workers=5) as ex:
            list(ex.map(_fetch_minute_yf, stale_min))

    def _fetch_with_today(ticker: str) -> pd.DataFrame:
        df = _cached_fetch(ticker)
        if ticker.endswith(".KS") or ticker.endswith(".KQ"):
            # 한국 주식: KIS API 실시간 바 (yfinance 사용 안 함)
            minute_df = _fetch_kr_realtime_bar(ticker)
        else:
            # 미국 주식: yfinance 5분봉
            minute_df = _fetch_minute_yf(ticker)
        return _append_today_bar(df, minute_df)

    signals = scan_all(stocks_to_scan, _fetch_with_today)

    if not signals:
        logger.debug("급등주 신호 없음")
        return

    logger.info("신호 발생: %d개 종목", len(signals))
    for sig in signals:
        # 한국 주식은 KIS API 실시간 현재가로 교정 (yfinance 오류 방지)
        ticker = sig["ticker"]
        if KIS_APP_KEY and (ticker.endswith(".KS") or ticker.endswith(".KQ")):
            try:
                code = ticker.replace(".KS", "").replace(".KQ", "")
                real_price = KISTrader().get_current_price(code)["price"]
                if real_price and real_price > 0:
                    sig["current_price"] = float(real_price)
            except Exception as e:
                logger.debug("KIS 현재가 조회 실패 [%s]: %s", ticker, e)

        result = send_signal_alert(sig, growth_cash)

        # 매수 성공 시 ML 포지션 저장 (익절/손절/7일 청산 추적용)
        if result.get("status") == "ok" and result.get("qty", 0) > 0:
            is_us = not (ticker.endswith(".KS") or ticker.endswith(".KQ"))
            save_ml_position(
                ticker       = ticker,
                name         = sig.get("name", ticker),
                qty          = result["qty"],
                entry_price  = result["price"],
                avg_win      = sig["avg_win"],
                is_us        = is_us,
            )


# ─────────────────────────────────────────────
# 월간 리밸런싱
# ─────────────────────────────────────────────

def retrain_kr_models():
    """매일 07:30 실행 — KRX 유니버스 ML 모델 재학습."""
    now = datetime.now(KST)
    if not is_kr_trading_day(now.date()):
        return
    logger.info("KR ML 재학습 시작")
    try:
        from ml.trainer import retrain_daily
        results = retrain_daily(market="kr")
        ok   = sum(1 for v in results.values() if v)
        fail = len(results) - ok
        send_telegram(
            f"🤖 <b>KR ML 모델 재학습 완료</b>\n"
            f"성공: {ok}개 / 실패: {fail}개\n"
            f"{now.strftime('%Y-%m-%d %H:%M')} 기준"
        )
    except Exception as e:
        logger.error("KR ML 재학습 실패: %s", e)
        send_telegram(f"⚠️ KR ML 재학습 오류: {e}")


def retrain_us_models():
    """미국 증시 시작 직후 실행 — US 유니버스 ML 모델 재학습.
    22:30(서머) / 23:30(동절기) 양쪽에 스케줄 등록 후
    ET 09:30~10:00 창에서만 실제 실행."""
    import pytz
    eastern = pytz.timezone("America/New_York")
    now_et  = datetime.now(KST).astimezone(eastern)
    et_min  = now_et.hour * 60 + now_et.minute
    if not (9 * 60 + 30 <= et_min <= 10 * 60):
        return
    logger.info("US ML 재학습 시작 (미국장 시작 직후)")
    try:
        from ml.trainer import retrain_daily
        results = retrain_daily(market="us")
        ok   = sum(1 for v in results.values() if v)
        fail = len(results) - ok
        send_telegram(
            f"🤖 <b>US ML 모델 재학습 완료</b>\n"
            f"성공: {ok}개 / 실패: {fail}개\n"
            f"{datetime.now(KST).strftime('%Y-%m-%d %H:%M')} 기준"
        )
    except Exception as e:
        logger.error("US ML 재학습 실패: %s", e)
        send_telegram(f"⚠️ US ML 재학습 오류: {e}")


def run_monthly_rebalance():
    """매월 1일 오전 8시 30분 실행 — 안전자산 비중 재조정."""
    if not is_bot_active():
        return

    now = datetime.now(KST)
    if not is_kr_trading_day(now.date()):
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
    schedule.every().day.at("07:30").do(retrain_kr_models)
    schedule.every().day.at("08:00").do(send_morning_briefing)
    schedule.every().day.at("15:00").do(send_daily_summary)
    schedule.every().day.at("22:30").do(retrain_us_models)  # 서머타임 미국장 시작
    schedule.every().day.at("23:30").do(retrain_us_models)  # 동절기 미국장 시작

    # ML 급등주 5분 스캔 (한국장 + 미국장 자동 필터링)
    schedule.every(5).minutes.do(scan_growth_signals)

    # ML 포지션 자동 청산 (익절 / 7거래일 강제)
    schedule.every(5).minutes.do(check_ml_positions)

    # 월간 리밸런싱 (매월 1일 08:30)
    schedule.every().day.at("08:30").do(
        lambda: run_monthly_rebalance() if datetime.now(KST).day == 1 else None
    )

    logger.info(
        "등록 완료: 07:30 KR재학습 / 08:00 모닝브리핑 / 15:00 일일리포트 / "
        "22:30·23:30 US재학습 / 5분 ML스캔+포지션청산 / 매월 1일 08:30 리밸런싱"
    )

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
