"""
graph.py - LangGraph 기반 퀀트 자동매매 에이전트
노드 순서: 데이터수집 → 지표계산 → 신호감지 → 뉴스수집 → AI판단 → 리스크체크 → 알림전송 → 로그저장
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
    STOCKS, ACTIVE_STRATEGY, TAVILY_API_KEY, OPENAI_API_KEY,
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
    news:         List[str]            # 뉴스 헤드라인 목록
    news_summary: str                  # 뉴스 요약 (긍정/부정/중립)
    ai_decision:  str                  # "매수" / "매도" / "보류"
    ai_reason:    str                  # AI 판단 근거
    risk_warning: str                  # 리스크 경고 메시지
    log_entries:  List[str]            # 로그 항목 목록
    error:        str                  # 오류 메시지


# ─────────────────────────────────────────────
# 노드 1: 데이터 수집
# ─────────────────────────────────────────────
def node_fetch_data(state: TraderState) -> TraderState:
    """yfinance로 종목 OHLCV 데이터를 수집합니다."""
    ticker = state["ticker"]
    logger.info("[노드1] 데이터 수집: %s", ticker)
    try:
        df = fetch_ohlcv(ticker, period_years=1)
        state["ohlcv"]  = df
        state["error"]  = ""
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
    """이동평균선과 RSI를 계산합니다."""
    logger.info("[노드2] 지표 계산: %s", state["ticker"])
    if state.get("error"):
        return state
    try:
        s = ACTIVE_STRATEGY
        df = add_all_indicators(
            state["ohlcv"],
            short=s["short_window"],
            long=s["long_window"],
            rsi_period=s["rsi_period"],
        )
        df = detect_crossover(df, short=s["short_window"], long=s["long_window"])
        state["ohlcv"] = df
        logger.info("  → 지표 계산 완료 (MA%d/MA%d, RSI%d)",
                    s["short_window"], s["long_window"], s["rsi_period"])
    except Exception as e:
        state["error"] = str(e)
        logger.error("  → 지표 계산 실패: %s", e)
    return state


# ─────────────────────────────────────────────
# 노드 3: 신호 감지
# ─────────────────────────────────────────────
def node_detect_signal(state: TraderState) -> TraderState:
    """골든크로스/데드크로스/RSI 조건을 체크하여 신호를 결정합니다."""
    logger.info("[노드3] 신호 감지: %s", state["ticker"])
    if state.get("error"):
        return state
    try:
        df = generate_signals(state["ohlcv"], strategy=ACTIVE_STRATEGY)
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

        logger.info("  → 신호: %s | RSI=%.1f | MA%d=%.0f | MA%d=%.0f",
                    state["signal_type"], signal["rsi"],
                    ACTIVE_STRATEGY["short_window"], signal["ma_short"],
                    ACTIVE_STRATEGY["long_window"],  signal["ma_long"])
    except Exception as e:
        state["error"] = str(e)
        logger.error("  → 신호 감지 실패: %s", e)
    return state


def route_after_signal(state: TraderState) -> Literal["news", "end"]:
    """신호가 없거나 오류면 END, 신호 있으면 뉴스 수집으로 이동"""
    if state.get("error") or state.get("signal_type") == "none":
        return "end"
    return "news"


# ─────────────────────────────────────────────
# 노드 4: 뉴스 수집
# ─────────────────────────────────────────────
def node_fetch_news(state: TraderState) -> TraderState:
    """Tavily API로 해당 종목의 최신 뉴스 3개를 수집합니다."""
    logger.info("[노드4] 뉴스 수집: %s", state["stock_name"])
    state["news"] = []
    state["news_summary"] = "중립적"

    if not TAVILY_API_KEY:
        logger.warning("  → TAVILY_API_KEY 미설정, 뉴스 수집 건너뜀")
        return state

    try:
        from tavily import TavilyClient
        client  = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(
            query=f"{state['stock_name']} 주식 뉴스",
            max_results=3,
            search_depth="basic",
        )
        headlines = [r.get("title", "") for r in results.get("results", [])]
        state["news"] = headlines
        logger.info("  → 뉴스 %d개 수집: %s", len(headlines), headlines)
    except Exception as e:
        logger.warning("  → 뉴스 수집 실패: %s", e)

    return state


# ─────────────────────────────────────────────
# 노드 5: AI 판단
# ─────────────────────────────────────────────
def node_ai_decision(state: TraderState) -> TraderState:
    """GPT-4o로 기술적 신호 + 뉴스 센티먼트를 종합 분석하여 최종 판단합니다."""
    logger.info("[노드5] AI 판단: %s", state["stock_name"])

    if not OPENAI_API_KEY:
        logger.warning("  → OPENAI_API_KEY 미설정, AI 판단 건너뜀 → 기술 신호 그대로 사용")
        # API 키 없으면 기술적 신호를 그대로 사용
        if state["signal_type"] == "buy":
            state["ai_decision"] = "매수"
            state["ai_reason"]   = "기술적 신호(골든크로스 + RSI)에 기반한 매수 신호"
        elif state["signal_type"] in ("sell_full", "sell_partial"):
            state["ai_decision"] = "매도"
            state["ai_reason"]   = "기술적 신호에 기반한 매도 신호"
        else:
            state["ai_decision"] = "보류"
            state["ai_reason"]   = "명확한 신호 없음"
        return state

    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=OPENAI_API_KEY,
        )

        news_text = "\n".join(f"- {n}" for n in state["news"]) if state["news"] else "뉴스 없음"
        signal    = state["signal"]

        prompt = f"""당신은 퀀트 투자 전문가입니다. 아래 정보를 바탕으로 투자 판단을 내려주세요.

종목: {state['stock_name']}
날짜: {signal['date']}
신호 유형: {state['signal_type']}
RSI: {signal['rsi']}
단기 이평선(MA{ACTIVE_STRATEGY['short_window']}): {signal['ma_short']}
장기 이평선(MA{ACTIVE_STRATEGY['long_window']}): {signal['ma_long']}

최신 뉴스:
{news_text}

위 정보를 종합하여 다음 JSON 형식으로만 응답해주세요:
{{"decision": "매수" or "매도" or "보류", "reason": "판단 근거 2~3문장", "news_sentiment": "긍정적" or "부정적" or "중립적"}}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        content  = response.content.strip()

        # JSON 파싱
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        parsed = json.loads(content)
        state["ai_decision"]  = parsed.get("decision", "보류")
        state["ai_reason"]    = parsed.get("reason", "")
        state["news_summary"] = parsed.get("news_sentiment", "중립적")

        logger.info("  → AI 판단: %s | 뉴스 센티먼트: %s",
                    state["ai_decision"], state["news_summary"])

    except Exception as e:
        logger.error("  → AI 판단 실패: %s", e)
        # 실패 시 기술 신호 기반 기본 판단
        state["ai_decision"] = "매수" if state["signal_type"] == "buy" else "매도"
        state["ai_reason"]   = f"AI 분석 오류로 기술 신호 기반 판단 ({e})"

    return state


def route_after_ai(state: TraderState) -> Literal["risk", "end"]:
    """AI 판단이 보류면 END, 매수/매도면 리스크 체크로 이동"""
    if state.get("ai_decision") == "보류":
        logger.info("  → AI 판단: 보류 → 종료")
        return "end"
    return "risk"


# ─────────────────────────────────────────────
# 노드 6: 리스크 체크
# ─────────────────────────────────────────────
def node_risk_check(state: TraderState) -> TraderState:
    """
    당일 중복 신호 필터 및 연속 매수 신호 경고를 처리합니다.
    """
    logger.info("[노드6] 리스크 체크: %s", state["ticker"])
    today  = str(date.today())
    ticker = state["ticker"]
    state["risk_warning"] = ""

    # 당일 중복 신호 방지
    daily_key = f"{ticker}_{today}"
    prev_signals = _today_signals.get(daily_key, [])

    if state["signal_type"] in prev_signals:
        logger.warning("  → 당일 중복 신호 감지 — 주문 건너뜀")
        state["ai_decision"] = "보류"
        state["ai_reason"]  += " (당일 중복 신호 필터)"
        return state

    # 연속 매수 신호 경고
    if state["signal_type"] == "buy":
        _consecutive_buy_count[ticker] = _consecutive_buy_count.get(ticker, 0) + 1
        if _consecutive_buy_count[ticker] >= MAX_CONSECUTIVE_BUY:
            warning = f"연속 {_consecutive_buy_count[ticker]}회 매수 신호 — 과열 주의"
            state["risk_warning"] = warning
            logger.warning("  → %s", warning)
    else:
        # 매도 신호 발생 시 연속 카운트 초기화
        _consecutive_buy_count[ticker] = 0

    # 당일 신호 기록
    _today_signals.setdefault(daily_key, []).append(state["signal_type"])
    logger.info("  → 리스크 체크 통과")
    return state


# ─────────────────────────────────────────────
# 노드 7: 알림 전송 + 주문 실행
# ─────────────────────────────────────────────
def node_send_notification(state: TraderState) -> TraderState:
    """텔레그램 알림을 전송하고 KIS API로 실제 주문을 실행합니다."""
    logger.info("[노드7] 알림 전송: %s", state["stock_name"])
    signal     = state["signal"]
    stock_name = state["stock_name"]
    ai_reason  = state["ai_reason"]

    # 리스크 경고 알림
    if state.get("risk_warning"):
        ticker_digits = state["ticker"].replace(".KS", "")
        count = _consecutive_buy_count.get(state["ticker"], 0)
        warn_msg = build_warning_message(stock_name, count)
        send_telegram(warn_msg)

    # 신호 유형별 메시지 생성 및 전송
    if state["signal_type"] == "buy":
        msg = build_buy_message(stock_name, signal, state.get("news_summary", "중립적"), ai_reason)
    elif state["signal_type"] == "sell_full":
        msg = build_sell_full_message(stock_name, signal, ai_reason)
    else:
        msg = build_sell_partial_message(stock_name, signal, ai_reason)

    send_telegram(msg)
    logger.info("  → 텔레그램 알림 전송 완료")

    # ── 실제 주문 (KIS API 키가 설정된 경우에만) ──
    from config import KIS_APP_KEY, ORDER_AMOUNT
    if KIS_APP_KEY:
        try:
            trader      = KISTrader()
            stock_code  = state["ticker"].replace(".KS", "")
            price_info  = trader.get_current_price(stock_code)
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

            elif state["signal_type"] == "sell_partial":
                balance = trader.get_balance()
                holding = next((b for b in balance if b["stock_code"] == stock_code), None)
                if holding and holding["qty"] > 0:
                    qty = max(1, holding["qty"] // 2)
                    trader.sell(stock_code, qty)
                    logger.info("  → 분할 매도 50%%: %s %d주", stock_code, qty)

        except Exception as e:
            logger.error("  → 주문 실패: %s", e)
    else:
        logger.info("  → KIS_APP_KEY 미설정 — 주문 시뮬레이션 모드")

    return state


# ─────────────────────────────────────────────
# 노드 8: 로그 저장
# ─────────────────────────────────────────────
def node_save_log(state: TraderState) -> TraderState:
    """판단 근거와 최종 결과를 로그 파일에 저장합니다."""
    logger.info("[노드8] 로그 저장: %s", state["stock_name"])

    log_entry = {
        "timestamp":   datetime.now().isoformat(),
        "ticker":      state["ticker"],
        "stock_name":  state["stock_name"],
        "signal_type": state["signal_type"],
        "signal":      state["signal"],
        "ai_decision": state["ai_decision"],
        "ai_reason":   state["ai_reason"],
        "news":        state.get("news", []),
        "risk_warning": state.get("risk_warning", ""),
    }

    logger.info("=== 매매 로그 ===\n%s", json.dumps(log_entry, ensure_ascii=False, indent=2))
    return state


# ─────────────────────────────────────────────
# 그래프 구성
# ─────────────────────────────────────────────
def build_graph() -> StateGraph:
    """LangGraph 노드를 연결하여 실행 그래프를 반환합니다."""
    graph = StateGraph(TraderState)

    # 노드 등록
    graph.add_node("fetch_data",        node_fetch_data)
    graph.add_node("calc_indicators",   node_calc_indicators)
    graph.add_node("detect_signal",     node_detect_signal)
    graph.add_node("fetch_news",        node_fetch_news)
    graph.add_node("ai_decision",       node_ai_decision)
    graph.add_node("risk_check",        node_risk_check)
    graph.add_node("send_notification", node_send_notification)
    graph.add_node("save_log",          node_save_log)

    # 엣지 연결
    graph.set_entry_point("fetch_data")
    graph.add_edge("fetch_data",      "calc_indicators")
    graph.add_edge("calc_indicators", "detect_signal")

    # 신호 없으면 → END
    graph.add_conditional_edges(
        "detect_signal",
        route_after_signal,
        {"news": "fetch_news", "end": END},
    )

    graph.add_edge("fetch_news",   "ai_decision")

    # AI 보류면 → END
    graph.add_conditional_edges(
        "ai_decision",
        route_after_ai,
        {"risk": "risk_check", "end": END},
    )

    graph.add_edge("risk_check",        "send_notification")
    graph.add_edge("send_notification", "save_log")
    graph.add_edge("save_log",          END)

    return graph.compile()


def run_all_stocks():
    """모든 종목에 대해 에이전트를 순차 실행합니다."""
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
            "news":         [],
            "news_summary": "중립적",
            "ai_decision":  "보류",
            "ai_reason":    "",
            "risk_warning": "",
            "log_entries":  [],
            "error":        "",
        }

        try:
            result = app.invoke(initial_state)
            logger.info(">>> [%s] 완료: 신호=%s, AI=%s",
                        ticker, result["signal_type"], result["ai_decision"])
        except Exception as e:
            logger.error(">>> [%s] 에이전트 오류: %s", ticker, e)

    logger.info("\n퀀트 자동매매 에이전트 종료")


if __name__ == "__main__":
    run_all_stocks()
