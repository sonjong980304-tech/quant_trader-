"""
graph.py - LangGraph 기반 퀀트 자동매매 에이전트
노드 순서: 데이터수집 → 지표계산 → 신호감지 → 리스크체크 → 알림전송 → 로그저장
"""

import os
import json
import logging
import logging.handlers
from datetime import date, datetime
from typing import TypedDict, List, Dict, Any, Literal

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from data_fetcher import fetch_ohlcv
from indicators import add_all_indicators, detect_crossover
from strategy import generate_signals, get_latest_signal
from notifier import (
    send_telegram,
    build_buy_message,
    build_sell_full_message,
    build_sell_partial_message,
    build_warning_message,
)
from trader import KISTrader
from config import (
    STOCKS, MA_SHORT, MA_LONG, RSI_PERIOD, OPENAI_API_KEY,
    LOG_FILE, MAX_CONSECUTIVE_BUY, ORDER_AMOUNT,
)

# ─────────────────────────────────────────────
# 로거 설정
# ─────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
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
logger = logging.getLogger("quant_trader")

# 당일 처리된 신호 추적 (중복 방지)
_today_signals: Dict[str, List[str]] = {}
_consecutive_buy_count: Dict[str, int] = {}


# ─────────────────────────────────────────────
# 그래프 상태 타입 정의
# ─────────────────────────────────────────────
class TraderState(TypedDict):
    ticker:       str                  # 현재 처리 중인 종목 코드
    stock_name:   str                  # 종목명 (한글)
    ohlcv:        Any                  # OHLCV DataFrame
    signal:       Dict[str, Any]       # 최신 신호 딕셔너리
    signal_type:  str                  # "buy" / "sell_full" / "sell_partial" / "none"
    risk_warning: str                  # 리스크 경고 메시지
    log_entries:  List[str]            # 로그 항목 목록
    error:        str                  # 오류 메시지


# ─────────────────────────────────────────────
# 노드 1: 데이터 수집
# ─────────────────────────────────────────────
def node_fetch_data(state: TraderState) -> TraderState:
    ticker = state["ticker"]
    logger.info("[노드1] 데이터 수집: %s", ticker)
    try:
        df = fetch_ohlcv(ticker, period_years=1)
        state["ohlcv"] = df
        state["error"] = ""
        logger.info("  → %d행 수집 완료 (%s ~ %s)", len(df),
                    df.index[0].date(), df.index[-1].date())
    except Exception as e:
        state["error"] = str(e)
        logger.error("  → 데이터 수집 실패: %s", e)
    return state


# ─────────────────────────────────────────────
# 노드 2: 지표 계산
# ─────────────────────────────────────────────
def node_calc_indicators(state: TraderState) -> TraderState:
    logger.info("[노드2] 지표 계산: %s", state["ticker"])
    if state.get("error"):
        return state
    try:
        df = add_all_indicators(
            state["ohlcv"], short=MA_SHORT, long=MA_LONG, rsi_period=RSI_PERIOD,
        )
        df = detect_crossover(df, short=MA_SHORT, long=MA_LONG)
        state["ohlcv"] = df
        logger.info("  → 지표 계산 완료 (MA%d/MA%d, RSI%d)", MA_SHORT, MA_LONG, RSI_PERIOD)
    except Exception as e:
        state["error"] = str(e)
        logger.error("  → 지표 계산 실패: %s", e)
    return state


# ─────────────────────────────────────────────
# 노드 3: 신호 감지
# ─────────────────────────────────────────────
def node_detect_signal(state: TraderState) -> TraderState:
    logger.info("[노드3] 신호 감지: %s", state["ticker"])
    if state.get("error"):
        return state
    try:
        df     = generate_signals(state["ohlcv"])
        signal = get_latest_signal(df)
        state["signal"] = signal

        if signal["buy"]:
            state["signal_type"] = "buy"
        elif signal["sell_full"]:
            state["signal_type"] = "sell_full"
        elif signal["sell_partial"]:
            state["signal_type"] = "sell_partial"
        else:
            state["signal_type"] = "none"

        principles = ", ".join(signal.get("buy_which") or signal.get("sell_which") or []) or "-"
        logger.info("  → 신호: %s (%s) | MA%d=%.0f | MA%d=%.0f | 거래량: %d",
                    state["signal_type"], principles,
                    MA_SHORT, signal["ma_short"],
                    MA_LONG,  signal["ma_long"],
                    signal.get("volume", 0))
    except Exception as e:
        state["error"] = str(e)
        logger.error("  → 신호 감지 실패: %s", e)
    return state


def route_after_signal(state: TraderState) -> Literal["risk", "end"]:
    if state.get("error") or state.get("signal_type") == "none":
        return "end"
    return "risk"


# ─────────────────────────────────────────────
# 노드 4: 리스크 체크
# ─────────────────────────────────────────────
def node_risk_check(state: TraderState) -> TraderState:
    logger.info("[노드4] 리스크 체크: %s", state["ticker"])
    today  = str(date.today())
    ticker = state["ticker"]
    state["risk_warning"] = ""

    daily_key    = f"{ticker}_{today}"
    prev_signals = _today_signals.get(daily_key, [])

    if state["signal_type"] in prev_signals:
        logger.warning("  → 당일 중복 신호 감지 — 주문 건너뜀")
        state["signal_type"] = "none"
        return state

    if state["signal_type"] == "buy":
        _consecutive_buy_count[ticker] = _consecutive_buy_count.get(ticker, 0) + 1
        if _consecutive_buy_count[ticker] >= MAX_CONSECUTIVE_BUY:
            warning = f"연속 {_consecutive_buy_count[ticker]}회 매수 신호 — 과열 주의"
            state["risk_warning"] = warning
            logger.warning("  → %s", warning)
    else:
        _consecutive_buy_count[ticker] = 0

    _today_signals.setdefault(daily_key, []).append(state["signal_type"])
    logger.info("  → 리스크 체크 통과")
    return state


def route_after_risk(state: TraderState) -> Literal["notify", "end"]:
    if state.get("signal_type") == "none":
        return "end"
    return "notify"


# ─────────────────────────────────────────────
# 노드 5: 알림 전송 + 주문 실행
# ─────────────────────────────────────────────
def node_send_notification(state: TraderState) -> TraderState:
    logger.info("[노드5] 알림 전송: %s", state["stock_name"])
    signal     = state["signal"]
    stock_name = state["stock_name"]

    if state.get("risk_warning"):
        warn_msg = build_warning_message(stock_name, _consecutive_buy_count.get(state["ticker"], 0))
        send_telegram(warn_msg)

    if state["signal_type"] == "buy":
        msg = build_buy_message(stock_name, signal)
    elif state["signal_type"] == "sell_full":
        msg = build_sell_full_message(stock_name, signal)
    else:
        msg = build_sell_partial_message(stock_name, signal)

    send_telegram(msg)
    logger.info("  → 텔레그램 알림 전송 완료")

    from config import KIS_APP_KEY
    if KIS_APP_KEY:
        try:
            trader        = KISTrader()
            stock_code    = state["ticker"].replace(".KS", "").replace(".KQ", "")
            price_info    = trader.get_current_price(stock_code)
            current_price = price_info["price"]

            if state["signal_type"] == "buy" and current_price > 0:
                qty = max(1, ORDER_AMOUNT // current_price)
                trader.buy(stock_code, qty)
                logger.info("  → 매수 주문: %s %d주 @ 시장가", stock_code, qty)

            elif state["signal_type"] == "sell_full":
                balance = trader.get_balance()
                holding = next((b for b in balance if b["stock_code"] == stock_code), None)
                if holding and holding["qty"] > 0:
                    trader.sell(stock_code, holding["qty"])
                    logger.info("  → 전량 매도: %s %d주", stock_code, holding["qty"])

        except Exception as e:
            logger.error("  → 주문 실패: %s", e)
    else:
        logger.info("  → KIS_APP_KEY 미설정 — 주문 시뮬레이션 모드")

    return state


# ─────────────────────────────────────────────
# 노드 6: 로그 저장
# ─────────────────────────────────────────────
def node_save_log(state: TraderState) -> TraderState:
    logger.info("[노드6] 로그 저장: %s", state["stock_name"])
    log_entry = {
        "timestamp":    datetime.now().isoformat(),
        "ticker":       state["ticker"],
        "stock_name":   state["stock_name"],
        "signal_type":  state["signal_type"],
        "signal":       state["signal"],
        "risk_warning": state.get("risk_warning", ""),
    }
    logger.info("=== 매매 로그 ===\n%s", json.dumps(log_entry, ensure_ascii=False, indent=2))
    return state


# ─────────────────────────────────────────────
# 그래프 구성
# ─────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(TraderState)

    graph.add_node("fetch_data",        node_fetch_data)
    graph.add_node("calc_indicators",   node_calc_indicators)
    graph.add_node("detect_signal",     node_detect_signal)
    graph.add_node("risk_check",        node_risk_check)
    graph.add_node("send_notification", node_send_notification)
    graph.add_node("save_log",          node_save_log)

    graph.set_entry_point("fetch_data")
    graph.add_edge("fetch_data",    "calc_indicators")
    graph.add_edge("calc_indicators", "detect_signal")

    graph.add_conditional_edges(
        "detect_signal",
        route_after_signal,
        {"risk": "risk_check", "end": END},
    )
    graph.add_conditional_edges(
        "risk_check",
        route_after_risk,
        {"notify": "send_notification", "end": END},
    )
    graph.add_edge("send_notification", "save_log")
    graph.add_edge("save_log",          END)

    return graph.compile()


def run_all_stocks():
    app = build_graph()
    logger.info("=" * 60)
    logger.info("퀀트 자동매매 에이전트 시작 (%s)", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    for ticker, stock_name in STOCKS.items():
        logger.info("\n>>> [%s] %s 처리 시작", ticker, stock_name)

        initial_state: TraderState = {
            "ticker":       ticker,
            "stock_name":   stock_name,
            "ohlcv":        None,
            "signal":       {},
            "signal_type":  "none",
            "risk_warning": "",
            "log_entries":  [],
            "error":        "",
        }

        try:
            result = app.invoke(initial_state)
            logger.info(">>> [%s] 완료: 신호=%s", ticker, result["signal_type"])
        except Exception as e:
            logger.error(">>> [%s] 에이전트 오류: %s", ticker, e)

    logger.info("\n퀀트 자동매매 에이전트 종료")


if __name__ == "__main__":
    run_all_stocks()
