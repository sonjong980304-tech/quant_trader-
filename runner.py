"""
runner.py - Reversion + Trend 슬롯분리 전략 스케줄러

스케줄:
  07:30   → retrain_kr_models()
  08:00   → send_morning_briefing()
  09:00   → execute_pending_orders("KR")
  15:00   → send_daily_summary()
  15:31   → scan_growth_signals_eod() : KR EOD ML 신호 스캔 → 익일 시초가 매수
"""

import schedule
import time
import threading
import logging
import logging.handlers
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta

import pandas as pd
import pytz
import yfinance as yf

from config import (
    STOCKS, MA_SHORT, MA_LONG, RSI_PERIOD, KIS_APP_KEY,
    LIVE_TRADING,
)
from position_manager import (
    _load_state, _check_activation, is_bot_active,
    save_ml_position, check_ml_positions,
    _init_legacy_tickers,
)
from data_fetcher import fetch_ohlcv, get_minute_data
from indicators import add_all_indicators, detect_crossover
from strategy import generate_signals, get_latest_signal
from trader import KISTrader
from notifier import send_telegram, build_daily_summary_message
from morning_briefer import send_morning_briefing
from signals.signal_graph import scan_all_graph
from market_calendar import is_kr_trading_day

KST      = pytz.timezone("Asia/Seoul")
LOG_FILE = "logs/trader.log"

# OHLCV 캐시: {ticker: (DataFrame, timestamp)}  — 30분 TTL
_OHLCV_CACHE: dict = {}
_CACHE_TTL   = 1800  # 초

# 분봉 캐시: {ticker: (DataFrame, timestamp)}  — 5분 TTL
_MINUTE_CACHE: dict = {}
_MINUTE_TTL   = 300  # 초

# 당일 알림 전송 기록: {ticker: "YYYY-MM-DD"}
_alerted_today: dict = {}


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

    # KIS 실시간 잔고에서 실제 평균단가 조회 (entry_price 보정용)
    kis_avg: dict = {}
    if KIS_APP_KEY:
        try:
            balance = KISTrader().get_balance()
            for h in balance:
                if h.get("avg_price") and h.get("qty", 0) > 0:
                    kis_avg[h["stock_code"]] = float(h["avg_price"])
        except Exception as e:
            logger.warning("잔고 평균단가 조회 실패: %s", e)

    logger.info("일일 기술적 분석 리포트 전송 시작")
    send_telegram(
        f"📊 <b>일일 기술적 분석 리포트</b> (보유 {len(ml_positions)}종목)\n"
        f"{now.strftime('%Y-%m-%d %H:%M')} 기준"
    )

    for ticker, pos_info in ml_positions.items():
        stock_name = pos_info.get("name", ticker)
        # KIS 실제 평균단가 우선 반영
        code = ticker.replace(".KS", "").replace(".KQ", "")
        if code in kis_avg:
            pos_info = {**pos_info, "avg_price": kis_avg[code]}
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

def _run_paper_evaluate_kr(trade_day: bool = False):
    """페이퍼 포지션 TP/SL 평가. trade_day=True 일 때만 trade_days 1일 증가."""
    try:
        from paper_trader import evaluate_positions_auto
        evaluate_positions_auto(trade_day=trade_day)
    except Exception as e:
        logger.warning("[Paper] KR 포지션 평가 실패: %s", e)


def _run_paper_evaluate_kr_eod():
    """15:30 EOD 전용 — trade_days 1일 증가 + TP/SL 체크. 거래일에만 실행."""
    if not is_kr_trading_day(datetime.now(KST).date()):
        return
    _run_paper_evaluate_kr(trade_day=True)


def _run_paper_daily_report_kr():
    """15:35 KR 페이퍼 트레이딩 일일 리포트 (한국 증시 마감 직후, 평일만)."""
    from datetime import datetime
    if datetime.now().weekday() >= 5:   # 토(5)·일(6) 스킵
        return
    try:
        from paper_trader import daily_report
        daily_report(market="KR")
    except Exception as e:
        logger.warning("[Paper] KR 일일 리포트 실패: %s", e)


_us_report_sent_date = ""   # 중복 발송 방지 (서머타임·동절기 이중 등록 대응)


def _run_paper_daily_report_us():
    """05:30 / 06:30 US 페이퍼 트레이딩 일일 리포트 (미국 증시 마감 직후, 하루 1회)."""
    global _us_report_sent_date
    today = datetime.now().strftime("%Y-%m-%d")
    if _us_report_sent_date == today:
        return
    _us_report_sent_date = today
    try:
        from paper_trader import daily_report
        daily_report(market="US")
    except Exception as e:
        logger.warning("[Paper] US 일일 리포트 실패: %s", e)


def _run_paper_weekly_summary():
    """일요일 20:00 페이퍼 트레이딩 주차별 집계."""
    try:
        from paper_trader import weekly_summary
        weekly_summary()
    except Exception as e:
        logger.warning("[Paper] 주차별 집계 실패: %s", e)


def _run_paper_entry_update(market: str):
    """장 시작 직후 호출 — entry_price=None 포지션에 실제 시초가 확정.
    KR: 09:05 KST, US: ET 09:35 (KST 22:35 서머/23:35 동절기).
    """
    if market == "KR" and not is_kr_trading_day(datetime.now(KST).date()):
        return
    try:
        from paper_trader import update_entry_prices
        update_entry_prices(market)
    except Exception as e:
        logger.warning("[Paper] 시초가 업데이트 실패 (%s): %s", market, e)


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
    """미국 장 여부 (서머타임 자동 감지).
    ET 기준 월~금 09:30~16:00 를 KST로 역산해 판단.
    KST 기준 요일이 아닌 ET 기준 요일로 체크해야 일요일 밤(US 월요일) 누락을 방지.
    """
    import pytz
    eastern = pytz.timezone("America/New_York")
    now_et  = datetime.now(KST).astimezone(eastern)
    if now_et.weekday() >= 5:   # ET 기준 주말
        return False
    et_min = now_et.hour * 60 + now_et.minute
    return 9 * 60 + 30 <= et_min < 16 * 60


# ─────────────────────────────────────────────
# EOD 신호 스캔 (15:31) — Train/Serve Skew 해소 (GATE B)
# ─────────────────────────────────────────────

def scan_growth_signals_eod():
    """
    15:31 실행 — Close 확정 후 EOD 일봉으로 신호 탐지, 익일 시초가 예약.

    학습 피처(EOD 완성 일봉)와 예측 피처를 동일하게 맞춰 Train/Serve Skew를 해소.
    신호는 즉시 매수하지 않고 pending_orders에 등록 → execute_pending_orders("KR")가
    09:00 시초가에 실행.
    """
    if not _check_activation():
        return
    if not is_kr_trading_day(datetime.now(KST).date()):
        return

    logger.info("EOD 신호 스캔 시작 (15:31)")

    from signals.krx_universe import get_krx_backtest_universe
    stocks_to_scan = get_krx_backtest_universe(top_n=200)
    if not stocks_to_scan:
        stocks_to_scan = STOCKS

    held = set(_load_state().get("ml_positions", {}).keys())
    # LIVE_TRADING=False 시 페이퍼 포지션의 ticker도 중복 신호 차단
    if not LIVE_TRADING:
        try:
            from paper_trader import _load as _pt_load, POS_PATH as _PT_POS
            _paper_pos = _pt_load(_PT_POS, {})
            held = held | {v["ticker"] for v in _paper_pos.values()}
        except Exception as _e:
            logger.debug("페이퍼 포지션 ticker 로드 실패 (무시): %s", _e)
    stocks_to_scan = {t: n for t, n in stocks_to_scan.items() if t not in held}

    # 완성된 일봉만 사용 — 분봉 합성 없음 (학습 시점과 동일한 피처 생성)
    def _fetch_eod_only(ticker: str) -> pd.DataFrame:
        return _cached_fetch(ticker)

    signals = scan_all_graph(stocks_to_scan, _fetch_eod_only)
    if not signals:
        logger.info("EOD 신호 없음")
        return

    logger.info("EOD 신호 %d개 발생", len(signals))

    # trend 에이전트 레짐 필터: KOSPI 종가 > KOSPI MA200
    def _kospi_above_ma200() -> bool:
        try:
            import yfinance as yf
            _k = yf.download("^KS11", period="250d", interval="1d", progress=False, auto_adjust=True)
            if _k.empty:
                return True
            _c = float(_k["Close"].iloc[-1])
            _m200 = float(_k["Close"].rolling(200).mean().iloc[-1])
            return _c > _m200
        except Exception:
            return True  # 데이터 취득 실패 시 허용

    _trend_allowed = _kospi_above_ma200()

    total_asset = 0.0
    if KIS_APP_KEY:
        try:
            total_asset = _get_total_asset(KISTrader())
        except Exception:
            pass
    growth_cash = total_asset if total_asset > 0 else 10_000_000

    from pending_orders import add_pending_order
    for sig in signals:
        if sig.get("agent") == "trend" and not _trend_allowed:
            logger.info("[레짐 필터] trend 차단 — KOSPI ≤ MA200: %s", sig.get("ticker"))
            continue
        ticker = sig["ticker"]
        is_us  = not (ticker.endswith(".KS") or ticker.endswith(".KQ"))
        atr    = sig.get("atr", 0)

        # Risk Parity 수량 (총 자산 1% 리스크 기준)
        if atr > 0 and total_asset > 0:
            rp_qty = max(1, int((total_asset * 0.01) / (2.0 * atr * (1400 if is_us else 1))))
        else:
            close  = max(sig.get("current_price", 10000), 1)
            rp_qty = max(1, int(growth_cash * 0.02 / close))

        code    = ticker.replace(".KS", "").replace(".KQ", "")
        ml_meta = {
            "avg_win":  sig["avg_win"],
            "avg_loss": sig["avg_loss"],
            "atr":      atr,
            "is_us":    is_us,
            "win_prob": sig["win_prob"],
            "agent":    sig.get("agent", ""),
        }

        if not LIVE_TRADING:
            logger.warning("[LIVE_TRADING=False] EOD 페이퍼: %s 승률=%.1f%%",
                           ticker, sig["win_prob"] * 100)
            try:
                from paper_trader import log_paper_signal, is_circuit_breaker_active, can_add_position
                _agent = sig.get("agent", "reversion")
                if not is_circuit_breaker_active() and can_add_position(_agent):
                    # 익일 시초가 진입 — EOD 종가는 eod_close로 저장, entry_price는 09:05에 확정
                    _ep_eod = float(sig.get("current_price") or 0.0)
                    log_paper_signal(
                        ticker           = ticker,
                        name             = sig.get("name", ticker),
                        agent            = _agent,
                        trigger_types    = sig.get("trigger_types", []),
                        win_prob         = sig["win_prob"],
                        avg_win          = sig["avg_win"],
                        avg_loss         = sig["avg_loss"],
                        rr               = sig.get("risk_reward", 0.0),
                        regime_prob      = sig.get("regime_prob"),
                        regime_pass      = True,
                        entry_price      = None,
                        actual_price     = None,
                        position_size_pct= 0.0,
                        kelly_fraction   = None,
                        auc_at_signal    = sig.get("model_auc"),
                        eod_close        = _ep_eod,
                    )
            except Exception as _pe:
                logger.warning("[Paper] EOD 신호 기록 실패: %s", _pe)
            send_telegram(
                f"📋 [페이퍼/EOD] <b>{sig.get('name', ticker)} ({ticker})</b>\n"
                f"승률 {sig['win_prob']*100:.1f}% | 손익비 {sig.get('risk_reward',0):.2f}\n"
                f"<i>GATE A·B·C 미통과 — 실주문 차단</i>"
            )
            continue

        from paper_trader import can_add_position as _cap_live
        _agent_live = sig.get("agent", "reversion")
        if not _cap_live(_agent_live):
            logger.info("[슬롯 초과] %s agent=%s — 확인 스킵", ticker, _agent_live)
            continue

        from pending_confirmations import add_confirmation
        from notifier import send_buy_confirmation_keyboard

        conf_id = add_confirmation(
            ticker  = ticker,
            name    = sig.get("name", ticker),
            qty     = rp_qty,
            code    = code,
            is_us   = is_us,
            ml_meta = ml_meta,
            note    = f"EOD ML 신호 | 승률={sig['win_prob']*100:.1f}%",
        )
        send_buy_confirmation_keyboard(
            f"🔔 [EOD 매수 신호] <b>{sig.get('name', ticker)} ({ticker})</b>\n"
            f"승률 {sig['win_prob']*100:.1f}% | 손익비 {sig.get('risk_reward', 0):.2f} | 수량 {rp_qty}주\n"
            f"익일 09:00 시초가 매수 예약 — 확인하시겠습니까?",
            conf_id,
        )

    # ── trend 에이전트 스캔 (ADX≥25 + MA정배열 + 거래량>1.3x) ──────────────────
    if _trend_allowed:
        try:
            from trend_agent import compute_indicators as _ti_compute
            from paper_trader import (
                log_paper_signal as _lps,
                can_add_position as _cap,
                is_circuit_breaker_active as _icba,
            )
            if not _icba():
                for _tk, _tn in stocks_to_scan.items():
                    if not _cap("trend"):
                        break
                    try:
                        _df = _fetch_eod_only(_tk)
                        if _df is None or len(_df) < 210:
                            continue
                        _df = _ti_compute(_df)
                        _r = _df.iloc[-1]
                        if any(pd.isna(_r.get(c)) for c in ["adx", "ma5", "ma20", "ma60", "ma200", "atr"]):
                            continue
                        if float(_r["adx"]) < 25:
                            continue
                        if not (float(_r["ma5"]) > float(_r["ma20"]) > float(_r["ma60"]) > float(_r["ma200"])):
                            continue
                        _vma = float(_r.get("vol_ma20") or 0)
                        if _vma == 0 or float(_r["Volume"]) < _vma * 1.3:
                            continue
                        _atr   = float(_r["atr"])
                        _close = float(_r["Close"])
                        _adx   = float(_r["adx"])
                        if not LIVE_TRADING:
                            _lps(
                                ticker        = _tk,
                                name          = _tn,
                                agent         = "trend",
                                trigger_types = ["trend_entry"],
                                entry_price   = None,
                                eod_close     = _close,
                                atr_at_entry  = _atr,
                            )
                            logger.info("[Trend] 페이퍼 신호: %s ADX=%.1f", _tk, _adx)
                            send_telegram(
                                f"📈 [페이퍼/Trend] <b>{_tn} ({_tk})</b>\n"
                                f"ADX={_adx:.1f} | ATR={_atr:.0f} | MA정배열 ✅\n"
                                f"익일 09:00 시초가 진입 예약"
                            )
                        else:
                            from pending_confirmations import add_confirmation
                            from notifier import send_buy_confirmation_keyboard
                            _rp_qty = max(1, int((total_asset * 0.01) / (2.0 * _atr))) if _atr > 0 and total_asset > 0 else max(1, int(growth_cash * 0.02 / _close))
                            _conf_id = add_confirmation(
                                ticker  = _tk,
                                name    = _tn,
                                qty     = _rp_qty,
                                code    = _tk.replace(".KS", "").replace(".KQ", ""),
                                is_us   = False,
                                ml_meta = {"atr": _atr, "agent": "trend", "avg_win": 0.0, "avg_loss": 0.0, "win_prob": 0.0, "is_us": False},
                                note    = f"Trend 신호 | ADX={_adx:.1f}",
                            )
                            send_buy_confirmation_keyboard(
                                f"🔔 [EOD Trend] <b>{_tn} ({_tk})</b>\n"
                                f"ADX={_adx:.1f} | ATR={_atr:.0f} | MA정배열 ✅\n"
                                f"익일 09:00 시초가 매수 예약 — 확인하시겠습니까?",
                                _conf_id,
                            )
                            logger.info("[Trend] 실매매 확인 요청: %s ADX=%.1f", _tk, _adx)
                    except Exception as _te:
                        logger.debug("trend 종목 스캔 실패 %s: %s", _tk, _te)
        except Exception as _trend_err:
            logger.warning("trend 신호 스캔 오류: %s", _trend_err)


# ─────────────────────────────────────────────
# US EOD 신호 스캔 (미국 장 마감 직후)
# ─────────────────────────────────────────────

def scan_growth_signals_eod_us():
    """
    05:20 / 06:20 실행 — 미국 장 마감(ET 16:00) 후 20분 뒤 EOD 일봉으로 신호 탐지, 당일 밤 시초가 예약.
    서머타임(ET 16:00 = KST 05:00) / 동절기(ET 16:00 = KST 06:00) 양쪽에 등록,
    내부에서 ET 16:00~17:00 창에서만 실행 (이중 등록 중복 방지).
    S&P 500 전체(503종목) 스캔.
    """
    if not _check_activation():
        return
    eastern = pytz.timezone("America/New_York")
    now_et  = datetime.now(KST).astimezone(eastern)
    if now_et.weekday() >= 5:
        return
    et_min = now_et.hour * 60 + now_et.minute
    if not (16 * 60 <= et_min < 17 * 60):
        return

    logger.info("US EOD 신호 스캔 시작 (%s ET)", now_et.strftime("%H:%M"))

    from signals.us_universe import get_us_candidates
    stocks_to_scan = get_us_candidates(top_n=503)
    if not stocks_to_scan:
        logger.info("US 유니버스 스크리닝 결과 없음")
        return

    held = set(_load_state().get("ml_positions", {}).keys())
    if not LIVE_TRADING:
        try:
            from paper_trader import _load as _pt_load, POS_PATH as _PT_POS
            _paper_pos = _pt_load(_PT_POS, {})
            held = held | {v["ticker"] for v in _paper_pos.values()}
        except Exception as _e:
            logger.debug("페이퍼 포지션 ticker 로드 실패 (무시): %s", _e)
    stocks_to_scan = {t: n for t, n in stocks_to_scan.items() if t not in held}

    def _fetch_eod_only(ticker: str) -> pd.DataFrame:
        return _cached_fetch(ticker)

    signals = scan_all_graph(stocks_to_scan, _fetch_eod_only)
    if not signals:
        logger.info("US EOD 신호 없음")
        return

    logger.info("US EOD 신호 %d개 발생", len(signals))

    total_asset = 0.0
    if KIS_APP_KEY:
        try:
            total_asset = _get_total_asset(KISTrader())
        except Exception:
            pass
    growth_cash = total_asset if total_asset > 0 else 10_000_000

    for sig in signals:
        ticker = sig["ticker"]
        atr    = sig.get("atr", 0)

        if atr > 0 and total_asset > 0:
            risk_usd = (total_asset * 0.01) / 1400
            rp_qty   = max(1, int(risk_usd / (2.0 * atr)))
        else:
            close  = max(sig.get("current_price", 10), 1)
            rp_qty = max(1, int((growth_cash * 0.02) / (close * 1400)))

        ml_meta = {
            "avg_win":  sig["avg_win"],
            "avg_loss": sig["avg_loss"],
            "atr":      atr,
            "is_us":    True,
            "win_prob": sig["win_prob"],
            "agent":    sig.get("agent", ""),
        }

        if not LIVE_TRADING:
            logger.warning("[LIVE_TRADING=False] US EOD 페이퍼: %s 승률=%.1f%%",
                           ticker, sig["win_prob"] * 100)
            try:
                from paper_trader import log_paper_signal, is_circuit_breaker_active
                if not is_circuit_breaker_active():
                    _ep_eod = float(sig.get("current_price") or 0.0)
                    log_paper_signal(
                        ticker           = ticker,
                        name             = sig.get("name", ticker),
                        agent            = sig.get("agent", "eod"),
                        trigger_types    = sig.get("trigger_types", []),
                        win_prob         = sig["win_prob"],
                        avg_win          = sig["avg_win"],
                        avg_loss         = sig["avg_loss"],
                        rr               = sig.get("risk_reward", 0.0),
                        regime_prob      = sig.get("regime_prob"),
                        regime_pass      = True,
                        entry_price      = None,
                        actual_price     = None,
                        position_size_pct= 0.0,
                        kelly_fraction   = None,
                        auc_at_signal    = sig.get("model_auc"),
                        eod_close        = _ep_eod,
                    )
            except Exception as _pe:
                logger.warning("[Paper] US EOD 신호 기록 실패: %s", _pe)
            send_telegram(
                f"📋 [페이퍼/US EOD] <b>{sig.get('name', ticker)} ({ticker})</b>\n"
                f"승률 {sig['win_prob']*100:.1f}% | 손익비 {sig.get('risk_reward',0):.2f}\n"
                f"<i>GATE A·B·C 미통과 — 실주문 차단</i>"
            )
            continue

        from pending_confirmations import add_confirmation
        from notifier import send_buy_confirmation_keyboard

        conf_id = add_confirmation(
            ticker  = ticker,
            name    = sig.get("name", ticker),
            qty     = rp_qty,
            code    = ticker,
            is_us   = True,
            ml_meta = ml_meta,
            note    = f"US EOD ML 신호 | 승률={sig['win_prob']*100:.1f}%",
        )
        send_buy_confirmation_keyboard(
            f"🔔 [US EOD 매수 신호] <b>{sig.get('name', ticker)} ({ticker})</b>\n"
            f"승률 {sig['win_prob']*100:.1f}% | 손익비 {sig.get('risk_reward', 0):.2f} | 수량 {rp_qty}주\n"
            f"당일 22:30·23:30 시초가 매수 예약 — 확인하시겠습니까?",
            conf_id,
        )


# ─────────────────────────────────────────────
# 월간 리밸런싱
# ─────────────────────────────────────────────

def execute_pending_orders(market: str):
    """장 시작 시 예약 주문 실행. market: 'KR' | 'US'"""
    if market == "KR" and not is_kr_trading_day(datetime.now(KST).date()):
        return
    if market == "US":
        eastern = pytz.timezone("America/New_York")
        if datetime.now(KST).astimezone(eastern).weekday() >= 5:
            return

    from pending_orders import pop_pending_orders
    from notifier import send_telegram

    orders = pop_pending_orders(market)
    if not orders:
        return

    logger.info("[예약주문] %s 장 시작 — %d건 실행", market, len(orders))
    results = []
    for o in orders:
        ticker  = o["ticker"]
        code    = o["code"]
        qty     = o["qty"]
        action  = o["action"]
        is_us   = o["is_us"]
        try:
            from trader import KISTrader
            t = KISTrader()
            if is_us:
                price_info = t.get_us_current_price(code)
                price = price_info["price"]
                name  = price_info.get("name", ticker)
                if action == "BUY":
                    t.buy_us(code, qty)
                    from trade_logger import log_buy
                    log_buy(ticker, code, price, qty, strategy="예약매수")
                else:
                    t.sell_us(code, qty)
                    from trade_logger import log_sell
                    log_sell(ticker, price, qty=qty, notes="예약매도")
                price_str = f"${price:,.2f}"
            else:
                price_info = t.get_current_price(code)
                price = price_info["price"]
                name  = price_info.get("name", ticker)
                if action == "BUY":
                    t.buy(code, qty)
                    from trade_logger import log_buy
                    log_buy(ticker, name, price, qty, strategy="예약매수")
                else:
                    t.sell(code, qty)
                    from trade_logger import log_sell
                    log_sell(ticker, price, qty=qty, notes="예약매도")
                price_str = f"{price:,}원"

            # ML EOD 신호 주문이면 포지션 추적 등록 (익절/손절/7일 청산용)
            if action == "BUY" and o.get("ml_meta"):
                meta = o["ml_meta"]
                save_ml_position(
                    ticker      = ticker,
                    name        = name,
                    qty         = qty,
                    entry_price = float(price),
                    avg_win     = meta.get("avg_win", 0.07),
                    atr         = meta.get("atr", 0.0),
                    is_us       = is_us,
                )

            icon = "✅"
            msg_detail = f"{price_str} × {qty}주"
            logger.info("[예약주문] 완료 %s %s %s", action, ticker, msg_detail)
        except Exception as e:
            icon = "❌"
            msg_detail = str(e)
            logger.error("[예약주문] 실패 %s %s: %s", action, ticker, e)
        results.append(f"{icon} {action} {ticker} {qty}주 — {msg_detail}")

    send_telegram(
        f"📋 <b>예약 주문 실행 결과 ({market}장)</b>\n" +
        "\n".join(results)
    )


_retrain_retry_kr = None  # threading.Timer
_retrain_retry_us = None  # threading.Timer
_RETRAIN_RETRY_INTERVAL = 1800  # 30분


def is_retrain_day(today: date) -> bool:
    """오늘이 분기별 재학습일(1/4/7/10월 1일 또는 직후 첫 영업일)인지 판정."""
    from config import RETRAIN_SCHEDULE
    mmdd = today.strftime('%m-%d')
    if mmdd in RETRAIN_SCHEDULE:
        return True
    # 재학습일이 휴일/주말이었으면 직후 첫 영업일에 실행 (최대 7일 내)
    for s in RETRAIN_SCHEDULE:
        m, d = int(s.split('-')[0]), int(s.split('-')[1])
        try:
            sched_date = date(today.year, m, d)
        except ValueError:
            continue
        gap = (today - sched_date).days
        if 1 <= gap <= 7:
            # sched_date ~ today 사이(exclusive today)에 영업일이 없어야 함
            interim = [sched_date + timedelta(days=i) for i in range(1, gap)]
            if not any(is_kr_trading_day(d_) for d_ in interim):
                return True
    return False


def next_retrain_date(from_date: date | None = None) -> str:
    """다음 분기 재학습 예정일 (MM-DD 기준 가장 가까운 미래 날짜) 반환."""
    from config import RETRAIN_SCHEDULE
    today = from_date or datetime.now(KST).date()
    candidates = []
    for year_offset in (0, 1):
        for s in RETRAIN_SCHEDULE:
            m, d = int(s.split('-')[0]), int(s.split('-')[1])
            try:
                c = date(today.year + year_offset, m, d)
            except ValueError:
                continue
            if c > today:
                candidates.append(c)
    candidates.sort()
    return candidates[0].strftime('%Y-%m-%d') if candidates else '알 수 없음'


def _needs_retry(results: dict) -> bool:
    """전체 종목 중 절반 이상 실패(또는 0개)면 True."""
    total = len(results)
    if total == 0:
        return True
    fail = sum(1 for v in results.values() if not v)
    return fail / total > 0.5


def retrain_kr_models(_is_retry: bool = False):
    """분기별 재학습 (1/4/7/10월 1일 또는 직후 첫 영업일). 절반 이상 실패 시 30분마다 재시도."""
    global _retrain_retry_kr
    now = datetime.now(KST)
    today = now.date()
    if not _is_retry:
        if not is_kr_trading_day(today):
            return
        if not is_retrain_day(today):
            return  # 재학습일 아님 — 추론(EOD 스캔)만 계속
        logger.info("분기별 재학습일 확인: %s — 재학습 시작", today)
    label = "재시도" if _is_retry else "시작"
    logger.info("KR ML 재학습 %s", label)
    try:
        from ml.trainer import retrain_daily
        results = retrain_daily(market="kr")
        ok   = sum(1 for v in results.values() if v)
        fail = len(results) - ok

        if _needs_retry(results):
            logger.warning("KR ML 재학습 절반 이상 실패(%d/%d) → 30분 후 재시도", fail, len(results))
            send_telegram(
                f"⚠️ <b>KR ML 재학습 실패 과다</b> ({fail}/{len(results)})\n"
                f"30분 후 자동 재시도합니다."
            )
            if _retrain_retry_kr:
                _retrain_retry_kr.cancel()
            _retrain_retry_kr = threading.Timer(
                _RETRAIN_RETRY_INTERVAL, retrain_kr_models, kwargs={"_is_retry": True}
            )
            _retrain_retry_kr.daemon = True
            _retrain_retry_kr.start()
        else:
            if _retrain_retry_kr:
                _retrain_retry_kr.cancel()
                _retrain_retry_kr = None
            send_telegram(
                f"🤖 <b>KR ML 모델 재학습 완료</b>\n"
                f"성공: {ok}개 / 실패: {fail}개\n"
                f"{now.strftime('%Y-%m-%d %H:%M')} 기준"
            )
    except Exception as e:
        logger.error("KR ML 재학습 실패: %s", e)
        send_telegram(f"⚠️ KR ML 재학습 오류: {e}")


def retrain_us_models(_is_retry: bool = False):
    """미국 증시 시작 직후 실행 — US 유니버스 ML 모델 재학습.
    22:30(서머) / 23:30(동절기) 양쪽에 스케줄 등록 후
    ET 09:30~10:00 창에서만 실제 실행."""
    global _retrain_retry_us
    import pytz
    eastern = pytz.timezone("America/New_York")
    now_et  = datetime.now(KST).astimezone(eastern)
    et_min  = now_et.hour * 60 + now_et.minute
    if not _is_retry and not (9 * 60 + 30 <= et_min <= 10 * 60):
        return
    label = "재시도" if _is_retry else "시작"
    logger.info("US ML 재학습 %s", label)
    try:
        from ml.trainer import retrain_daily
        results = retrain_daily(market="us")
        ok   = sum(1 for v in results.values() if v)
        fail = len(results) - ok

        if _needs_retry(results):
            logger.warning("US ML 재학습 절반 이상 실패(%d/%d) → 30분 후 재시도", fail, len(results))
            send_telegram(
                f"⚠️ <b>US ML 재학습 실패 과다</b> ({fail}/{len(results)})\n"
                f"30분 후 자동 재시도합니다."
            )
            if _retrain_retry_us:
                _retrain_retry_us.cancel()
            _retrain_retry_us = threading.Timer(
                _RETRAIN_RETRY_INTERVAL, retrain_us_models, kwargs={"_is_retry": True}
            )
            _retrain_retry_us.daemon = True
            _retrain_retry_us.start()
        else:
            if _retrain_retry_us:
                _retrain_retry_us.cancel()
                _retrain_retry_us = None
            send_telegram(
                f"🤖 <b>US ML 모델 재학습 완료</b>\n"
                f"성공: {ok}개 / 실패: {fail}개\n"
                f"{datetime.now(KST).strftime('%Y-%m-%d %H:%M')} 기준"
            )
    except Exception as e:
        logger.error("US ML 재학습 실패: %s", e)
        send_telegram(f"⚠️ US ML 재학습 오류: {e}")


def run_monthly_rebalance():
    """폐기됨 — 안전자산 리밸런싱 없이 슬롯 분리 10+10 전략만 운용."""
    logger.info("run_monthly_rebalance: 폐기된 호출 (슬롯 분리 10+10 전략)")


def main():
    logger.info("스케줄러 시작")

    # 기존 보유 종목 기록 (최초 1회)
    _init_legacy_tickers()

    # B1 전략 교체 기록 — paper 카운트 리셋 (이미 등록된 경우 skip)
    try:
        from paper_trader import _load, META_PATH, register_logic_change
        _hist = _load(META_PATH, {}).get("logic_change_history", [])
        if not any("B1" in h.get("reason", "") for h in _hist):
            register_logic_change("B1: EOD 익일 시초가 진입 전략 교체 (slip 0.25%→0.05%)")
    except Exception as _e:
        logger.warning("로직 변경 등록 실패: %s", _e)


    schedule.clear()

    # 고정 스케줄
    schedule.every().day.at("07:30").do(retrain_kr_models)
    schedule.every().day.at("08:00").do(send_morning_briefing)
    schedule.every().day.at("09:00").do(lambda: execute_pending_orders("KR"))  # 한국장 시작
    schedule.every().day.at("09:05").do(lambda: _run_paper_entry_update("KR"))  # KR 시초가 확정
    schedule.every().day.at("15:00").do(send_daily_summary)
    schedule.every().day.at("15:30").do(_run_paper_evaluate_kr_eod)     # KR EOD — trade_days+1 + TP/SL
    schedule.every().day.at("15:35").do(_run_paper_daily_report_kr)   # KR 마감 직후
    # [DEPRECATED 2026-06-20] US 운용 폐기 — SoT §5: 국내(KRX)만 운용
    # schedule.every().day.at("05:20").do(scan_growth_signals_eod_us)
    # schedule.every().day.at("06:20").do(scan_growth_signals_eod_us)
    # schedule.every().day.at("22:30").do(retrain_us_models)
    # schedule.every().day.at("22:30").do(lambda: execute_pending_orders("US"))
    # schedule.every().day.at("22:35").do(lambda: _run_paper_entry_update("US"))
    # schedule.every().day.at("23:30").do(retrain_us_models)
    # schedule.every().day.at("23:30").do(lambda: execute_pending_orders("US"))
    # schedule.every().day.at("23:35").do(lambda: _run_paper_entry_update("US"))
    schedule.every().sunday.at("20:00").do(_run_paper_weekly_summary)

    # B1: EOD 익일 시초가 전략 전환으로 장중 5분 신호 스캔 비활성화

    # EOD 신호 스캔 15:31 — Close 확정 후 EOD 일봉 기준 신호 → 익일 시초가 예약 (GATE B)
    schedule.every().day.at("15:31").do(scan_growth_signals_eod)

    # ML 포지션 자동 청산 (익절 / 7거래일 강제)
    schedule.every(5).minutes.do(check_ml_positions)
    # 페이퍼 포지션 장중 TP/SL 평가 — 실매매봇과 동일 주기
    schedule.every(5).minutes.do(_run_paper_evaluate_kr)

    # 월간 리밸런싱 (매월 1일 08:30)
    schedule.every().day.at("08:30").do(
        lambda: run_monthly_rebalance() if datetime.now(KST).day == 1 else None
    )

    logger.info(
        "등록 완료: 07:30 KR재학습(분기별 1/4/7/10월) / 08:00 모닝브리핑 / 15:00 일일리포트 / "
        "15:30 KR EOD평가 / 15:35 KR페이퍼 / 5분 ML+페이퍼TP/SL / 매월 1일 08:30 리밸런싱"
    )

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
