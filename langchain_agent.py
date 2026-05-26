"""
langchain_agent.py - LangChain 기반 주식 어시스턴트

기존 gpt_agent.py의 툴들을 LangChain Tool 형식으로 래핑.
ConversationBufferWindowMemory로 대화 기록 유지.
"""

import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from config import OPENAI_API_KEY, STOCKS, MA_SHORT, MA_LONG
from config import (
    VOLUME_INCREASE_RATIO, VOLUME_SURGE_RATIO,
    VOLUME_SURGE_MINUTE_RATIO, VOLUME_LOOKBACK_DAYS,
)

logger = logging.getLogger(__name__)

# 유저별 AgentExecutor 인스턴스 저장
_agents: dict[int, AgentExecutor] = {}
_memories: dict[int, ConversationBufferWindowMemory] = {}


# ─────────────────────────────────────────────
# LangChain Tools (기존 함수 래핑)
# ─────────────────────────────────────────────

@tool
def get_naver_finance(identifier: str) -> str:
    """
    Naver 증권에서 한국 주식 재무정보(PER, PBR, ROE, EPS, 현재가 등)를 조회합니다.
    identifier: 종목명(예: '삼성전자') 또는 6자리 코드(예: '005930')
    """
    from naver_finance import get_financials
    return get_financials(identifier)


@tool
def get_yahoo_finance(ticker: str) -> str:
    """
    Yahoo Finance에서 미국 주식 재무지표(PER, EPS, ROE, 매출 등)를 조회합니다.
    ticker: Yahoo Finance 심볼 (예: 'AAPL', 'TSLA', 'NVDA')
    """
    from gpt_agent import _call_yahoo_finance
    return _call_yahoo_finance(ticker)


@tool
def get_naver_news(query: str, n: int = 5) -> str:
    """
    네이버 뉴스에서 종목 또는 키워드 관련 최신 뉴스를 검색합니다.
    query: 검색 키워드 (예: 'LG전자', '반도체 업황')
    n: 뉴스 건수 (기본 5, 최대 10)
    """
    from gpt_agent import _call_naver_news
    return _call_naver_news(query, min(max(1, n), 10))


@tool
def get_stock_signal(identifier: str) -> str:
    """
    특정 종목의 현재 기술적 지표(MA, RSI, 거래량)와 매수/매도 신호를 분석합니다.
    identifier: 종목명(예: 'LG전자') 또는 6자리 코드(예: '066570')
    """
    from gpt_agent import _call_stock_signal
    return _call_stock_signal(identifier)


@tool
def get_historical_price(identifier: str, date: str) -> str:
    """
    특정 날짜의 종목 종가를 조회합니다.
    identifier: 종목명 또는 6자리 코드
    date: 날짜 (예: '2025-01-15', '2025년 1월 15일')
    """
    from gpt_agent import _call_historical_price
    return _call_historical_price(identifier, date)


@tool
def get_account_balance() -> str:
    """현재 계좌 잔고와 보유 종목 현황을 조회합니다 (국내 + 미국주식)."""
    try:
        from trader import KISTrader
        from config import KIS_APP_KEY, IS_MOCK
        if not KIS_APP_KEY:
            return "KIS_APP_KEY 미설정 — 잔고 조회 불가"
        t       = KISTrader()
        balance = t.get_balance()
        cash    = t.get_available_cash()
        total   = t.get_total_eval_amt()
        mode    = "모의투자" if IS_MOCK else "실전투자"
        lines   = [f"[계좌 잔고 — {mode}]",
                   f"주문 가능 현금: {cash:,}원",
                   f"총평가금액(매도대금 포함): {total:,}원"]

        if balance:
            lines.append("\n[국내주식]")
            for h in balance:
                lines.append(
                    f"  • {h['name']} ({h['stock_code']}): {h['qty']}주 | "
                    f"평균 {h['avg_price']:,}원 | 평가손익 {h['eval_profit']:+,}원"
                )
        else:
            lines.append("  (국내 보유 종목 없음)")

        try:
            us_balance = t.get_us_balance()
            if us_balance:
                lines.append("\n[미국주식 — 통합증거금]")
                for h in us_balance:
                    lines.append(
                        f"  • {h['name']} ({h['symbol']}): {h['qty']}주 | "
                        f"평균 ${h['avg_price']:.2f} | 평가손익 ${h['eval_profit']:+.2f}"
                    )
            else:
                lines.append("  (미국주식 보유 없음)")
        except Exception as e:
            lines.append(f"  (미국주식 조회 실패: {e})")

        return "\n".join(lines)
    except Exception as e:
        return f"잔고 조회 실패: {e}"


@tool
def get_portfolio_status() -> str:
    """안전자산 포트폴리오(70%) 현황과 리밸런싱 필요 여부를 확인합니다."""
    try:
        from portfolio.safe_portfolio import format_rebalance_report
        from trader import KISTrader
        from config import KIS_APP_KEY
        holdings    = {}
        total_asset = 10_000_000
        if KIS_APP_KEY:
            t           = KISTrader()
            balance     = t.get_balance()
            total_asset = t.get_total_eval_amt()
            holdings    = {h["stock_code"]: {"qty": h["qty"]} for h in balance}
        return format_rebalance_report(holdings, total_asset)
    except Exception as e:
        return f"포트폴리오 조회 실패: {e}"


@tool
def set_conditional_order(
    stock_name: str,
    condition_type: str,
    condition_value: float,
    action: str,
    quantity: int,
) -> str:
    """
    조건부 주문을 등록합니다.
    condition_type: 'price_below' | 'price_above' | 'profit_above' | 'profit_below'
    action: 'buy' | 'sell' | 'sellall'
    """
    from gpt_agent import _call_set_conditional_order
    return _call_set_conditional_order(stock_name, condition_type, condition_value, action, quantity)


@tool
def list_conditional_orders() -> str:
    """등록된 조건부 주문 목록을 조회합니다."""
    from gpt_agent import _call_list_conditional_orders
    return _call_list_conditional_orders()


@tool
def cancel_conditional_order(order_id: str) -> str:
    """조건부 주문을 취소합니다. order_id='all'이면 전체 취소."""
    from gpt_agent import _call_cancel_conditional_order
    return _call_cancel_conditional_order(order_id)


# ─────────────────────────────────────────────
# 시스템 프롬프트
# ─────────────────────────────────────────────

def _build_system_prompt() -> str:
    try:
        from trader import KISTrader
        from config import KIS_APP_KEY
        if KIS_APP_KEY:
            t        = KISTrader()
            holdings = t.get_balance()
            cash     = t.get_available_cash()
            total    = t.get_total_eval_amt()
            pos_text = "\n".join(
                f"  - {h['name']} ({h['stock_code']}): {h['qty']}주 @ {h['avg_price']:,}원"
                for h in holdings
            ) or "  (보유 없음)"
            cash_text = f"  주문 가능 현금: {cash:,}원 / 총평가금액(매도대금 포함): {total:,}원"
        else:
            pos_text  = "  (KIS 미연결)"
            cash_text = "  (KIS 미연결)"
    except Exception:
        pos_text  = "  (조회 실패)"
        cash_text = "  (조회 실패)"

    stocks_text = "\n".join(f"  - {n} ({t})" for t, n in STOCKS.items()) or "  (없음)"

    return f"""당신은 퀀트 자동매매 시스템의 AI 어시스턴트입니다.
사용자의 매매 전략과 포트폴리오를 정확히 알고 있으며, 주식 관련 질문에 도움을 줍니다.
답변은 반드시 한국어로 하세요.

━━━ 포트폴리오 구조 ━━━
  안전자산 70%: QQQ 22.3% / 삼성전자 27.3% / TLT 0.2% / ACE KRX금현물 50.3%
  급등주 30%: XGBoost ML 모델 + 켈리 공식 포지션 사이징

━━━ 매매 전략 원칙 ━━━
  MA{MA_SHORT}/MA{MA_LONG} + 거래량 + 캔들 기반 (기존 전략 유지)
  급등주: 승률 ≥ 55% AND 손익비 ≥ 1.5 조건 충족 시 알림

━━━ 계좌 현황 ━━━
{cash_text}

━━━ 현재 보유 종목 ━━━
{pos_text}

━━━ 관심 종목 ━━━
{stocks_text}"""


# ─────────────────────────────────────────────
# Agent 빌더
# ─────────────────────────────────────────────

_TOOLS = [
    get_naver_finance,
    get_yahoo_finance,
    get_naver_news,
    get_stock_signal,
    get_historical_price,
    get_account_balance,
    get_portfolio_status,
    set_conditional_order,
    list_conditional_orders,
    cancel_conditional_order,
]


def _build_agent(user_id: int) -> AgentExecutor:
    llm = ChatOpenAI(
        model="gpt-4.1",
        api_key=OPENAI_API_KEY,
        temperature=0,
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10,  # 최근 10턴 유지
    )
    _memories[user_id] = memory

    prompt = ChatPromptTemplate.from_messages([
        ("system", _build_system_prompt()),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, _TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=_TOOLS,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5,
    )


def get_agent(user_id: int) -> AgentExecutor:
    """유저별 AgentExecutor 싱글턴 반환."""
    if user_id not in _agents:
        _agents[user_id] = _build_agent(user_id)
    return _agents[user_id]


# ─────────────────────────────────────────────
# 공개 인터페이스
# ─────────────────────────────────────────────

def ask(user_id: int, question: str) -> str:
    """
    LangChain 에이전트로 질문 처리.
    gpt_agent.ask()와 동일한 시그니처 — telegram_bot.py에서 교체 가능.
    """
    try:
        agent  = get_agent(user_id)
        result = agent.invoke({"input": question})
        answer = result.get("output", "")

        # Context Recall 평가 (툴 호출이 있었을 때만)
        if result.get("intermediate_steps"):
            ctx = "\n\n".join(
                str(step[1]) for step in result["intermediate_steps"]
            )
            score = _eval_recall(ctx, answer)
            return f"{answer}\n\n─────────────────\n📊 Context Recall: {score:.2f}"

        return answer
    except Exception as e:
        logger.error("LangChain 에이전트 오류: %s", e)
        return f"⚠️ 오류: {e}"


def clear_history(user_id: int):
    """대화 기록 초기화."""
    if user_id in _memories:
        _memories[user_id].clear()
    _agents.pop(user_id, None)


def _eval_recall(context: str, answer: str) -> float:
    """Context Recall 점수 평가 (0~1)."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        f"Context Recall = 컨텍스트로 근거를 찾을 수 있는 답변 내 주장 수 / 답변 내 전체 주장 수\n\n"
        f"[컨텍스트]\n{context[:2000]}\n\n[답변]\n{answer}\n\n"
        f"0.00~1.00 사이 숫자만 반환:"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        return round(min(max(float(resp.choices[0].message.content.strip()), 0.0), 1.0), 2)
    except Exception:
        return -1.0
