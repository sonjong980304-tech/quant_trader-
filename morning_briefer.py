"""
morning_briefer.py - 평일 오전 8시 자동 모닝 브리핑

흐름:
  1. Tavily 웹 검색으로 미국 증시 / 관심종목 뉴스 / 경제 캘린더 수집
  2. GPT-A(gpt-5.5): 4가지 질문에 대한 브리핑 생성
  3. GPT-B(gpt-5.4-mini): Context Recall 평가 (검색 결과를 얼마나 충실히 반영했는지)
  4. 브리핑 + 평가 결과를 텔레그램으로 전송
"""

import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict

import pytz
from openai import OpenAI
from tavily import TavilyClient
from langgraph.graph import StateGraph, END

from config import STOCKS, OPENAI_API_KEY
from notifier import send_telegram
from market_calendar import is_kr_trading_day

logger = logging.getLogger(__name__)
KST    = pytz.timezone("Asia/Seoul")


# ─────────────────────────────────────────────
# 웹 검색
# ─────────────────────────────────────────────

def _search(query: str, k: int = 5, days: int = 2) -> str:
    """Tavily 검색 결과를 문자열로 반환 (days: 최근 N일 이내 결과만)"""
    import os
    client  = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
    results = client.search(
        query,
        max_results=k,
        days=days,
        search_depth="advanced",
    )["results"]
    return "\n\n".join(
        f"[출처: {r['url']}]\n{r['content']}"
        for r in results
    )


def _get_holding_names() -> list[str]:
    """
    KIS API에서 실제 보유 종목명 조회.
    실패 시 config STOCKS 목록으로 폴백.
    """
    try:
        from config import KIS_APP_KEY
        if KIS_APP_KEY:
            from trader import KISTrader
            holdings = KISTrader().get_balance()
            names = [h["name"] for h in holdings if h.get("name")]
            if names:
                return names
    except Exception:
        pass
    # 폴백: 관심종목 전체
    return list(STOCKS.values())


def _get_us_indices() -> str:
    """yfinance로 S&P 500 / 나스닥 / 다우존스 최근 거래일 종가 및 등락률 직접 조회."""
    import yfinance as yf
    indices = [("S&P 500", "^GSPC"), ("나스닥 종합", "^IXIC"), ("다우존스", "^DJI")]
    lines = []
    for name, ticker in indices:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) >= 2:
                prev  = float(hist["Close"].iloc[-2])
                last  = float(hist["Close"].iloc[-1])
                chg   = (last - prev) / prev * 100
                date  = hist.index[-1].strftime("%Y-%m-%d")
                arrow = "▲" if chg >= 0 else "▼"
                lines.append(f"{name} ({date}): {last:,.2f} ({arrow}{abs(chg):.2f}%)")
        except Exception as e:
            lines.append(f"{name}: 조회 실패 ({e})")
    return "\n".join(lines) if lines else "지수 조회 실패"


def _gather_context() -> dict:
    """미국 증시 / 보유종목 뉴스 / 경제 캘린더를 병렬 검색 + 실제 지수 데이터 직접 조회"""
    now        = datetime.now(KST)
    from datetime import timedelta
    us_date    = (now - timedelta(days=1)).strftime("%B %d %Y")
    kst_today  = now.strftime("%Y-%m-%d")

    # 보유 종목 기준 뉴스 (없으면 관심종목 전체)
    holding_names = _get_holding_names()
    stock_list    = " OR ".join(holding_names)

    queries = {
        "us_market":     f"US stock market close {us_date} S&P500 Nasdaq Composite Dow Jones recap results",
        "stock_news":    f"{kst_today} ({stock_list}) 주식 뉴스",
        "econ_calendar": f"economic calendar {now.strftime('%B %Y')} upcoming events this week CPI FOMC schedule",
    }

    context = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_search, q): key for key, q in queries.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                context[key] = future.result()
            except Exception as e:
                logger.warning("검색 실패 (%s): %s", key, e)
                context[key] = "검색 결과 없음"

    # 실제 지수 수치 직접 조회 (AI hallucination 방지)
    context["us_indices"] = _get_us_indices()

    return context


# ─────────────────────────────────────────────
# GPT-A: 브리핑 생성
# ─────────────────────────────────────────────

def _generate_briefing(context: dict) -> str:
    """검색 결과를 바탕으로 4가지 질문에 답하는 브리핑 생성"""
    client     = OpenAI(api_key=OPENAI_API_KEY)
    stock_list = ", ".join(_get_holding_names())

    user_prompt = f"""아래 검색 결과를 바탕으로 다음 4가지 질문에 답해줘.

=== 미국 주요 지수 실제 데이터 (반드시 이 수치를 사용할 것) ===
{context.get("us_indices", "")}

=== 미국 증시 뉴스/시황 ===
{context.get("us_market", "")}

=== 관심종목 뉴스 ===
{context.get("stock_news", "")}

=== 경제 캘린더 ===
{context.get("econ_calendar", "")}

---

1. 간밤 미국 증시 마감시황을 요약해줘. 반드시 위 "미국 주요 지수 실제 데이터"의 수치를 그대로 사용해서 다우존스, S&P500, 나스닥 각각의 등락률을 포함하고, 상승/하락의 원인을 한 가지만 짚어줘.
2. 위의 결과를 바탕으로, 오늘 한국 주식 시장의 개장 분위기를 긍정, 중립, 부정 중 하나로 판단하고 그 이유를 설명해줘.
3. 내 현재 관심종목({stock_list})과 직접 관련된 핵심 뉴스가 있다면 한 개씩만 요약해줘. 뉴스가 없으면 "특이사항 없음"으로 보고해.
4. 오늘을 기준으로 발표가 예정된 시장에 영향을 줄 수 있는 주요 경제 지표나 이벤트가 있다면 시간(한국 시간 기준)과 함께 알려줘.
"""

    resp = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "system", "content": "당신은 전문 금융 시황 브리핑 어시스턴트입니다. 간결하고 정확하게 답변하세요."},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


# ─────────────────────────────────────────────
# GPT-B: Context Recall 평가
# ─────────────────────────────────────────────

def _evaluate_context_recall(briefing: str, context: dict) -> tuple:
    """
    검색 컨텍스트 대비 브리핑의 Context Recall을 평가.
    - 점수(0.0~1.0): 컨텍스트 핵심 정보 중 브리핑에 반영된 비율
    - 평가 코멘트: 잘 반영된 점 / 누락된 점
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 컨텍스트가 너무 길면 잘라서 평가에 사용
    ctx_text = (
        f"[미국 증시]\n{context.get('us_market', '')[:1500]}\n\n"
        f"[경제 캘린더]\n{context.get('econ_calendar', '')[:800]}"
    )

    eval_prompt = f"""당신은 RAG 품질 평가 전문가입니다. Context Recall을 평가해주세요.

Context Recall이란: 검색된 컨텍스트의 핵심 정보 중 실제 답변에 반영된 비율입니다.

=== 검색된 컨텍스트 ===
{ctx_text}

=== 생성된 브리핑 ===
{briefing}

---
위 브리핑이 컨텍스트의 핵심 정보를 얼마나 충실히 반영했는지 평가해주세요.

반드시 아래 형식으로만 답변하세요:
점수: [0.0~1.0 사이의 숫자]
잘된 점: [한 문장]
아쉬운 점: [한 문장, 없으면 "없음"]
"""

    resp = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0,
    )
    content = resp.choices[0].message.content.strip()

    # 점수 파싱
    score = 0.0
    for line in content.splitlines():
        if line.startswith("점수:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass

    return score, content


# ─────────────────────────────────────────────
# LangGraph 브리핑 품질 재시도 루프
# score < 0.7이면 최대 3회까지 브리핑 재생성
# ─────────────────────────────────────────────

class _BriefingState(TypedDict):
    context:     dict
    briefing:    str
    score:       float
    evaluation:  str
    retry_count: int


def _node_generate(state: _BriefingState) -> _BriefingState:
    if state["retry_count"] > 0:
        logger.info("  브리핑 재생성 (시도 %d회)...", state["retry_count"] + 1)
    briefing = _generate_briefing(state["context"])
    return {**state, "briefing": briefing}


def _node_evaluate(state: _BriefingState) -> _BriefingState:
    score, evaluation = _evaluate_context_recall(state["briefing"], state["context"])
    logger.info("  Context Recall 점수: %.2f (시도 %d회)", score, state["retry_count"] + 1)
    return {**state, "score": score, "evaluation": evaluation, "retry_count": state["retry_count"] + 1}


def _route_quality(state: _BriefingState) -> str:
    """점수 ≥ 0.7이거나 3회 시도했으면 전송, 아니면 재생성."""
    if state["score"] >= 0.7 or state["retry_count"] >= 3:
        return "send"
    return "regenerate"


def _build_briefing_graph():
    g = StateGraph(_BriefingState)
    g.add_node("generate", _node_generate)
    g.add_node("evaluate", _node_evaluate)
    g.set_entry_point("generate")
    g.add_edge("generate", "evaluate")
    g.add_conditional_edges("evaluate", _route_quality, {
        "send":       END,
        "regenerate": "generate",
    })
    return g.compile()


_briefing_graph = _build_briefing_graph()


# ─────────────────────────────────────────────
# 메인 진입점
# ─────────────────────────────────────────────

def send_morning_briefing():
    """평일 오전 8시 자동 실행 — 모닝 브리핑 수집·생성·품질 재시도·텔레그램 전송"""
    now = datetime.now(KST)
    if not is_kr_trading_day(now.date()):
        return

    logger.info("모닝 브리핑 시작 (%s)", now.strftime("%Y-%m-%d %H:%M"))

    try:
        # 1. 컨텍스트 수집 (병렬 검색)
        logger.info("  웹 검색 중...")
        context = _gather_context()

        # 2. LangGraph 품질 재시도 루프 (생성 → 평가 → 재생성, 최대 3회)
        logger.info("  브리핑 생성 + 품질 평가 시작...")
        result = _briefing_graph.invoke({
            "context":     context,
            "briefing":    "",
            "score":       0.0,
            "evaluation":  "",
            "retry_count": 0,
        })
        briefing   = result["briefing"]
        score      = result["score"]
        evaluation = result["evaluation"]
        retries    = result["retry_count"]

        # 3. 텔레그램 전송
        retry_note = f" (재생성 {retries}회)" if retries > 1 else ""
        send_telegram(
            f"🌅 <b>모닝 브리핑</b> {now.strftime('%Y-%m-%d %H:%M')}{retry_note}"
        )
        send_telegram(briefing)
        send_telegram(
            f"📋 <b>브리핑 품질 평가 (Context Recall)</b>\n{evaluation}"
        )

        logger.info("  모닝 브리핑 완료 (recall=%.2f, 시도=%d회)", score, retries)

    except Exception as e:
        logger.error("모닝 브리핑 실패: %s", e)
        send_telegram(f"⚠️ 모닝 브리핑 오류: {e}")
