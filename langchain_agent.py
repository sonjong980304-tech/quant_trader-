"""
langchain_agent.py - LangGraph ReAct 기반 주식 어시스턴트

AgentExecutor → create_react_agent (langgraph.prebuilt) 전환.
MemorySaver checkpointer로 thread_id(유저별) 대화 이력 분리 관리.
clear_history()는 generation 카운터를 증가시켜 새 thread로 전환.
"""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from config import OPENAI_API_KEY, STOCKS, MA_SHORT, MA_LONG

logger = logging.getLogger(__name__)

# MemorySaver: thread_id 기반으로 유저별 대화 이력 분리
_checkpointer      = MemorySaver()
_react_agent       = None
# clear_history() 시 generation 증가 → 새 thread_id로 전환
_thread_generation: dict[int, int] = {}


# ─────────────────────────────────────────────
# LangChain Tools
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
    stock_code: str = "",
) -> str:
    """
    조건부 주문을 등록합니다.
    condition_type: 'price_below' | 'price_above' | 'profit_above' | 'profit_below'
    action: 'buy' | 'sell' | 'sellall'
    stock_code: 6자리 종목코드 (알면 반드시 입력 — 삼성전자=005930, SK하이닉스=000660, NAVER=035420)
    """
    from gpt_agent import _call_set_conditional_order
    return _call_set_conditional_order(stock_name, condition_type, condition_value, action, quantity, stock_code)


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


@tool
def list_trade_records(status: str = "all") -> str:
    """
    매매 이력 CSV를 조회합니다.
    status: 'open'(미청산), 'closed'(청산완료), 'all'(전체)
    """
    from trade_logger import _read_all
    rows = _read_all()
    if status == "open":
        rows = [r for r in rows if not r.get("exit_date")]
    elif status == "closed":
        rows = [r for r in rows if r.get("exit_date")]
    if not rows:
        return f"({status}) 매매 이력 없음"
    lines = [f"총 {len(rows)}건:"]
    for r in rows[-20:]:  # 최근 20건
        pnl = f" | 손익 {r['pnl_pct']}%" if r.get("pnl_pct") else ""
        lines.append(
            f"[{r['trade_id']}] {r['entry_date']} {r['ticker']} {r['name']} "
            f"{r['qty']}주 진입가={r['entry_price']}{pnl}"
        )
    return "\n".join(lines)


@tool
def edit_trade_record(trade_id: str, field: str, value: str) -> str:
    """
    매매 이력 CSV의 특정 항목을 수정합니다.
    trade_id: 수정할 거래 ID(8자리) 또는 종목코드(예: 005930.KS) 또는 종목명(예: 삼성전자).
              종목코드/종목명을 주면 가장 최근 거래를 자동 선택합니다.
    field: 수정할 컬럼명 (entry_price, exit_price, qty, entry_date, exit_date, strategy, notes, win_prob, name 등)
    value: 새로운 값
    수정 불가 컬럼: trade_id, ticker, side
    """
    from trade_logger import _read_all, _write_all
    IMMUTABLE = {"trade_id", "ticker", "side"}
    if field in IMMUTABLE:
        return f"⚠️ '{field}'는 수정 불가 컬럼입니다."
    rows = _read_all()
    # 8자리 ID 직접 매칭
    target = next((r for r in rows if r["trade_id"] == trade_id), None)
    # ID 미발견 시 → 종목코드 또는 종목명으로 검색 (가장 최근 거래)
    if not target:
        key = trade_id.strip()
        candidates = [
            r for r in rows
            if r.get("ticker", "").upper() == key.upper()
            or r.get("name", "") == key
        ]
        if not candidates:
            sample = "\n".join(
                f"  [{r['trade_id']}] {r['entry_date']} {r['ticker']} {r['name']}"
                for r in rows[-5:]
            )
            return f"⚠️ '{trade_id}'를 찾을 수 없습니다. 최근 거래:\n{sample}"
        target = sorted(candidates, key=lambda r: r.get("entry_date", ""))[-1]
    old_val = target.get(field, "")
    target[field] = value
    # pnl 재계산 (entry_price 또는 exit_price 변경 시)
    if field in ("entry_price", "exit_price") and target.get("exit_price") and target.get("entry_price"):
        try:
            ep = float(target["entry_price"])
            xp = float(target["exit_price"])
            qty = int(target.get("qty", 0) or 0)
            target["pnl_pct"]    = round((xp - ep) / ep * 100, 2)
            target["pnl_amount"] = round((xp - ep) * qty, 0)
            target["win"]        = 1 if target["pnl_pct"] > 0 else 0
        except Exception:
            pass
    _write_all(rows)
    return (
        f"✅ [{trade_id}] {target['ticker']} {target['name']}\n"
        f"  {field}: '{old_val}' → '{value}'"
        + (f"\n  손익 재계산: {target.get('pnl_pct')}%" if field in ("entry_price", "exit_price") else "")
    )


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

━━━ 툴 사용 원칙 (반드시 준수) ━━━
  현재 주가 / 재무지표(PER·PBR·ROE·EPS 등) 한국 주식 → get_naver_finance 호출
  현재 주가 / 재무지표 미국 주식 → get_yahoo_finance 호출
  특정 날짜의 과거 주가 → get_historical_price 호출
  기술적 분석·매수매도 신호 → get_stock_signal 호출
  뉴스 검색 → get_naver_news 호출
  계좌 잔고·보유 종목 → get_account_balance 호출
  매매 이력 조회 → list_trade_records 호출
  매매 이력 수정 (진입가/청산가/수량/메모 등 변경 요청 시) → edit_trade_record 호출
    ※ trade_id를 모를 경우 종목명(예: "삼성전자")이나 종목코드(예: "005930.KS")를 trade_id에 전달하면 자동 검색
  ※ 학습 데이터(training knowledge)로 주가·재무 수치를 절대 답변하지 마세요.
     반드시 툴을 호출해 실시간 데이터를 가져오세요.

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
# LangGraph ReAct Agent
# ─────────────────────────────────────────────

_TOOLS = [
    get_naver_finance,
    get_yahoo_finance,
    get_naver_news,
    get_stock_signal,
    list_trade_records,
    edit_trade_record,
    get_historical_price,
    get_account_balance,
    get_portfolio_status,
    set_conditional_order,
    list_conditional_orders,
    cancel_conditional_order,
]


def _thread_id(user_id: int) -> str:
    """유저 ID + generation으로 thread_id 생성. clear_history() 시 generation 증가."""
    gen = _thread_generation.get(user_id, 0)
    return f"{user_id}_{gen}"


def _get_react_agent():
    """create_react_agent 싱글턴 반환 (MemorySaver 공유)."""
    global _react_agent
    if _react_agent is None:
        llm = ChatOpenAI(
            model="gpt-5.5",
            api_key=OPENAI_API_KEY,
            temperature=0,
        )
        _react_agent = create_react_agent(
            model=llm,
            tools=_TOOLS,
            prompt=_build_system_prompt(),
            checkpointer=_checkpointer,
        )
    return _react_agent


def get_agent(user_id: int):
    """유저별 agent 반환 (단일 인스턴스, thread_id로 대화 이력 분리)."""
    return _get_react_agent()


# ─────────────────────────────────────────────
# 공개 인터페이스
# ─────────────────────────────────────────────

def ask(user_id: int, question: str) -> str:
    """
    LangGraph ReAct 에이전트로 질문 처리.
    thread_id = f"{user_id}_{generation}" 으로 유저별 대화 이력 분리.
    """
    try:
        agent  = _get_react_agent()
        config = {"configurable": {"thread_id": _thread_id(user_id)}}
        result = agent.invoke(
            {"messages": [("human", question)]},
            config=config,
        )
        messages = result.get("messages", [])
        answer   = messages[-1].content if messages else ""

        # Context Recall 평가 (툴 호출이 있었을 때만)
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_msgs:
            ctx   = "\n\n".join(m.content for m in tool_msgs)
            score = _eval_recall(ctx, answer)
            return f"{answer}\n\n─────────────────\n📊 Context Recall: {score:.2f}"

        return answer
    except Exception as e:
        logger.error("LangGraph 에이전트 오류: %s", e)
        return f"⚠️ 오류: {e}"


def clear_history(user_id: int):
    """대화 기록 초기화 — generation을 증가시켜 새 thread로 전환."""
    _thread_generation[user_id] = _thread_generation.get(user_id, 0) + 1
    logger.info("유저 %d 대화 이력 초기화 (thread_id: %s)", user_id, _thread_id(user_id))


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
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=10,
            temperature=0,
        )
        return round(min(max(float(resp.choices[0].message.content.strip()), 0.0), 1.0), 2)
    except Exception:
        return -1.0
