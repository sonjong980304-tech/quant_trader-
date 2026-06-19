"""
morning_briefer.py - 평일 오전 8시 자동 모닝 브리핑

흐름:
  1. Tavily 웹 검색으로 미국 증시 / 관심종목 뉴스 / 경제 캘린더 수집
  2. GPT-A(gpt-5.4-mini): 4가지 질문에 대한 브리핑 생성
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
        "econ_calendar": f"economic calendar {now.strftime('%B %Y')} upcoming events this week global CPI FOMC BOE ECB Bank of England rate decision schedule",
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

    # 경제 이벤트 날짜 검증 (날짜 없는 항목은 Tavily 추가 검색)
    context = _find_event_dates(context)

    return context


def _find_event_dates(context: dict) -> dict:
    """
    econ_calendar + us_market에서 오늘 이후 예정 이벤트 추출.
    날짜가 불명확한 항목은 Tavily로 총 최대 3회 추가 검색.
    결과를 context["confirmed_events"]에 저장.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    today  = datetime.now(KST).strftime("%Y-%m-%d")

    from datetime import timedelta
    deadline = (datetime.now(KST) + timedelta(days=30)).strftime("%Y-%m-%d")

    extract_prompt = f"""아래 텍스트에서 경제 이벤트를 모두 찾으세요.

[경제 캘린더]
{context.get('econ_calendar', '')[:1200]}

[미국 증시 뉴스]
{context.get('us_market', '')[:800]}

규칙:
- 날짜가 있든 없든 언급된 경제 이벤트를 모두 포함하세요
  (예: BOE 금리 결정, 미국 CPI 발표, FOMC 회의, 영국 기준금리 결정 등)
- 이미 지난 이벤트(오늘 {today} 이전)는 제외
- 오늘로부터 한 달({deadline}) 이후 이벤트도 제외
- 이벤트명은 반드시 한국어로 작성 (영어 금지)
- 날짜가 명확하면 YYYY-MM-DD, 불명확하거나 없으면 미확인
- 텍스트에 이벤트가 전혀 언급되지 않았을 때만 "없음" 출력

반드시 아래 형식으로만 답하세요 (한 줄에 하나씩):
이벤트명: YYYY-MM-DD 또는 미확인
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": extract_prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("이벤트 추출 실패: %s", e)
        context["confirmed_events"] = "이벤트 조회 실패"
        return context

    if raw == "없음":
        context["confirmed_events"] = "예정된 이벤트 없음"
        return context

    # 파싱
    events: dict[str, str] = {}
    for line in raw.splitlines():
        if ": " in line:
            name, date = line.split(": ", 1)
            events[name.strip()] = date.strip()

    # 날짜 미확인 항목 → Tavily 추가 검색 (총 최대 3회)
    search_count = 0
    month = datetime.now(KST).strftime("%B %Y")
    for name in list(events.keys()):
        if events[name] != "미확인" or search_count >= 3:
            continue
        try:
            result = _search(f"{name} scheduled date {month}", k=3, days=14)
            date_resp = client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[{"role": "user", "content": (
                    f"다음 텍스트에서 '{name}'의 예정 날짜를 YYYY-MM-DD 형식으로 추출하세요. "
                    f"명확한 날짜가 없으면 '미확인'만 출력:\n{result[:600]}"
                )}],
                temperature=0,
            )
            found = date_resp.choices[0].message.content.strip()
            events[name] = found
            search_count += 1
            logger.info("  [이벤트 날짜] %s → %s", name, found)
        except Exception as e:
            logger.warning("  [이벤트 날짜 검색 실패] %s: %s", name, e)

    lines = [
        f"- {name}: {'날짜 미확인' if date == '미확인' else date}"
        for name, date in events.items()
    ]
    context["confirmed_events"] = "\n".join(lines) if lines else "예정된 이벤트 없음"
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

=== 경제 이벤트 (날짜 검증 완료 — 반드시 이 목록만 사용할 것) ===
{context.get("confirmed_events", "")}

---

1. 간밤 미국 증시 마감시황을 요약해줘. 반드시 위 "미국 주요 지수 실제 데이터"의 수치를 그대로 사용해서 다우존스, S&P500, 나스닥 각각의 등락률을 포함하고, 상승/하락의 원인을 한 가지만 짚어줘.
2. 위의 결과를 바탕으로, 오늘 한국 주식 시장의 개장 분위기를 긍정, 중립, 부정 중 하나로 판단하고 그 이유를 설명해줘.
3. 내 현재 관심종목({stock_list})과 직접 관련된 핵심 뉴스가 있다면 한 개씩만 요약해줘. 뉴스가 없으면 "특이사항 없음"으로 보고해.
4. 위 "경제 이벤트" 목록을 바탕으로 예정된 이벤트를 한국 시간 기준으로 알려줘. 날짜가 "날짜 미확인"인 항목은 반드시 날짜를 알 수 없다고 명시해줘. 이 목록에 없는 이벤트는 절대 포함하지 말 것.
"""

    resp = client.chat.completions.create(
        model="gpt-5.4-mini",
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
    항목별 체크리스트로 브리핑 품질 평가.
    항목1: 지수 수치·등락률 (0.3점)
    항목2: 상승/하락 원인   (0.3점)
    항목3: 미래 이벤트·날짜 (0.2점)
    항목4: 관심종목 뉴스    (0.2점, 뉴스 없으면 패스)
    반환: (정규화 점수 0.0~1.0, 세부 결과 문자열)
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    stock_news = context.get("stock_news", "")
    has_stock_news = bool(stock_news.strip()) and stock_news != "검색 결과 없음"

    ctx_text = (
        f"[미국 증시 뉴스]\n{context.get('us_market', '')[:1500]}\n\n"
        f"[경제 이벤트 (날짜 검증 완료)]\n{context.get('confirmed_events', '')}\n\n"
        f"[미국 지수 실제 데이터]\n{context.get('us_indices', '')}\n\n"
        f"[관심종목 뉴스]\n{stock_news[:600]}"
    )

    item4_instruction = (
        "항목4 (최대 0.2점): 관심종목 관련 뉴스가 브리핑에 언급됐는가?"
        if has_stock_news else
        "항목4: 관심종목 뉴스가 없으므로 자동으로 '패스' 처리. 반드시 '패스'로 답하세요."
    )

    eval_prompt = f"""당신은 모닝 브리핑 품질 평가 전문가입니다. 아래 컨텍스트를 기준으로 브리핑을 항목별로 채점하세요.

=== 검색된 컨텍스트 ===
{ctx_text}

=== 생성된 브리핑 ===
{briefing}

---
채점 기준:

항목1 (최대 0.3점): 미국 3대 지수(S&P500, 나스닥, 다우존스) 수치와 등락률이 모두 정확히 포함됐는가?
항목2 (최대 0.3점): 증시 상승 또는 하락의 원인이 최소 한 가지 명확히 언급됐는가?
항목3 (최대 0.2점): "경제 이벤트 (날짜 검증 완료)" 목록의 이벤트들이 브리핑에 올바르게 포함됐는가? 날짜 미확인 항목을 브리핑에서 날짜 불명확하게 표시했으면 정답 처리. 목록이 비어있거나 "예정된 이벤트 없음"이면 0.2점 만점 부여.
{item4_instruction}

반드시 아래 형식으로만 답변하세요 (숫자는 소수점 둘째 자리):
항목1: [0.00~0.30]
항목2: [0.00~0.30]
항목3: [0.00~0.20]
항목4: [0.00~0.20 또는 패스]
항목1감점: [만점 미달 시 한 문장 이유, 만점이면 없음]
항목2감점: [만점 미달 시 한 문장 이유, 만점이면 없음]
항목3감점: [만점 미달 시 한 문장 이유, 만점이면 없음]
항목4감점: [만점 미달 시 한 문장 이유, 만점이거나 패스면 없음]
"""

    resp = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0,
    )
    raw_content = resp.choices[0].message.content.strip()

    # ── 파싱 ──────────────────────────────────────────────────────────
    scores  = {"항목1": 0.0, "항목2": 0.0, "항목3": 0.0, "항목4": None}
    reasons = {"항목1": "", "항목2": "", "항목3": "", "항목4": ""}

    for line in raw_content.splitlines():
        for key in scores:
            if line.startswith(f"{key}감점:"):
                val = line.split(":", 1)[1].strip()
                if val != "없음":
                    reasons[key] = val
            elif line.startswith(f"{key}:"):
                val = line.split(":", 1)[1].strip()
                if val == "패스":
                    scores[key] = "패스"
                else:
                    try:
                        scores[key] = round(float(val), 2)
                    except ValueError:
                        pass

    # ── 정규화 점수 계산 ──────────────────────────────────────────────
    max_possible = 0.3 + 0.3 + 0.2 + (0.2 if scores["항목4"] != "패스" else 0.0)
    raw_sum  = sum(v for v in scores.values() if isinstance(v, float))
    normalized = round(raw_sum / max_possible, 2) if max_possible > 0 else 0.0

    # ── 텔레그램용 포맷 ──────────────────────────────────────────────
    LABELS = {
        "항목1": ("지수 수치·등락률", 0.3),
        "항목2": ("상승/하락 원인  ", 0.3),
        "항목3": ("경제 지표·이벤트", 0.2),
        "항목4": ("관심종목 뉴스  ", 0.2),
    }

    lines = []
    for key, (label, max_val) in LABELS.items():
        val = scores[key]
        if val == "패스":
            line = f"├ {label}: 패스"
        else:
            score_str = f"{val:.2f}/{max_val:.1f}"
            reason    = f" — {reasons[key]}" if reasons[key] else ""
            line      = f"├ {label}: {score_str}{reason}"
        lines.append(line)

    lines.append(f"└ 총점: {raw_sum:.2f}/{max_possible:.1f}")
    evaluation = "\n".join(lines)

    return normalized, evaluation


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
