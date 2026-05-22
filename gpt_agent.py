"""
gpt_agent.py - GPT 기반 주식 질의응답 에이전트

ask(user_id, question) → str

동작 방식:
  1. 전략/포지션/종목 정보를 시스템 프롬프트에 주입
  2. GPT가 재무 수치가 필요하다고 판단하면 get_naver_finance 툴 자동 호출
  3. 이미 알고 있는 정보(전략, 원칙 등)는 툴 없이 바로 답변
"""

import json
import logging
from openai import OpenAI

from config import (
    OPENAI_API_KEY, STOCKS, MA_SHORT, MA_LONG,
    VOLUME_INCREASE_RATIO, VOLUME_SURGE_RATIO, VOLUME_SURGE_MINUTE_RATIO,
    VOLUME_LOOKBACK_DAYS,
)

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

_histories: dict[int, list[dict]] = {}
_MAX_HISTORY = 20   # 유저당 최대 보관 메시지 수

_pending_orders: dict[int, dict] = {}   # 확인 대기 중인 주문
_CONFIRM_WORDS = {"응", "네", "맞아", "맞습니다", "진행", "진행해", "확인", "예", "yes", "ㅇ", "ㅇㅇ", "해줘", "실행", "실행해", "고"}
_DENY_WORDS    = {"아니", "아니오", "아니요", "취소", "no", "ㄴ", "그만", "안해", "안 해", "하지마", "하지 마"}

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_naver_finance",
            "description": (
                "Naver 증권에서 특정 종목의 재무정보를 실시간으로 조회합니다. "
                "PER, PBR, ROE, EPS, BPS, 매출액, 영업이익, 당기순이익, 부채비율, "
                "시가배당률 등 구체적인 수치가 필요하거나 확신이 없을 때 사용하세요. "
                "매매 전략·원칙·파라미터처럼 이미 알고 있는 정보는 이 툴 없이 바로 답변하세요."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": (
                            "조회할 종목의 이름(예: '삼성전자') 또는 "
                            "6자리 종목코드(예: '005930')"
                        ),
                    }
                },
                "required": ["identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_conditional_order",
            "description": (
                "조건부 주문을 등록합니다. 특정 가격 조건이나 수익률 조건 충족 시 자동으로 매수/매도가 실행됩니다. "
                "'7만원 아래로 떨어지면 사줘', '+10% 되면 팔아줘', '-5% 되면 손절해줘' 같은 요청에 사용하세요."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_name": {"type": "string", "description": "종목명 (예: '삼성전자')"},
                    "condition_type": {
                        "type": "string",
                        "enum": ["price_below", "price_above", "profit_above", "profit_below"],
                        "description": (
                            "price_below: 현재가 < 기준가 / price_above: 현재가 > 기준가 / "
                            "profit_above: 수익률 > X% (익절) / profit_below: 수익률 < X% (손절)"
                        ),
                    },
                    "condition_value": {
                        "type": "number",
                        "description": "기준값. 가격 조건이면 원 단위, 수익률 조건이면 % 숫자 (예: 10 → +10%)",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["buy", "sell", "sellall"],
                        "description": "buy: 매수 / sell: 일부 매도 / sellall: 전량 매도",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "주문 수량. sellall이면 0",
                    },
                },
                "required": ["stock_name", "condition_type", "condition_value", "action", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_conditional_orders",
            "description": "등록된 조건부 주문 목록을 조회합니다. '조건부 주문 뭐 걸어놨어?' 같은 질문에 사용하세요.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_conditional_order",
            "description": "조건부 주문을 취소합니다. 주문 ID를 지정하거나 '전부 취소'로 모두 삭제할 수 있습니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "취소할 주문 ID (list_conditional_orders에서 확인). 전체 취소면 'all'",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_trade",
            "description": (
                "사용자가 명시적으로 매수/매도 주문을 요청할 때 호출합니다. "
                "실제 주문 전 사용자에게 확인을 받기 위해 주문 내용을 제안합니다. "
                "'삼성전자 10주 사줘', '현대차 팔아줘', '전량 매도해줘' 같은 직접적인 주문 요청에만 사용하세요. "
                "분석·조회 질문에는 절대 사용하지 마세요."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["buy", "sell", "sellall"],
                        "description": "buy: 매수 / sell: 일부 매도 / sellall: 전량 매도",
                    },
                    "stock_code": {
                        "type": "string",
                        "description": "6자리 종목코드 (예: '005930'). .KS/.KQ 제외",
                    },
                    "stock_name": {
                        "type": "string",
                        "description": "종목명 (예: '삼성전자')",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "주문 수량. sellall이면 0",
                    },
                },
                "required": ["action", "stock_code", "stock_name", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_yahoo_finance",
            "description": (
                "Yahoo Finance에서 미국 주식의 재무지표를 조회합니다. "
                "PER, PBR, EPS, ROE, 매출액, 영업이익, 순이익, 부채비율, 시가총액, 배당수익률 등 "
                "시계열 재무제표 데이터도 포함합니다. "
                "미국 주식(AAPL, TSLA, NVDA 등) 재무 질문에 사용하세요."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Yahoo Finance 티커 심볼 (예: 'AAPL', 'TSLA', 'NVDA', 'MSFT')",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_naver_news",
            "description": (
                "네이버 뉴스 API로 특정 종목 또는 키워드 관련 최신 뉴스를 검색합니다. "
                "'최근 뉴스', '요즘 이슈', '뭔 일 있어?', '기사 찾아줘' 같은 질문에 사용하세요."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 종목명 또는 키워드 (예: 'LG전자', '반도체 업황')",
                    },
                    "n": {
                        "type": "integer",
                        "description": "가져올 뉴스 건수 (기본 5, 최대 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_signal",
            "description": (
                "특정 종목의 현재 기술적 지표와 매수/매도 원칙별 충족 여부를 상세히 분석합니다. "
                "현재가, MA5, MA20, RSI, 거래량 비율, 캔들 타입, "
                "각 매수/매도 원칙의 조건 충족 여부(실제 수치 포함)를 반환합니다. "
                "'왜 매수 신호가 없어?', '오늘 신호 어때?', '조건 분석해줘' 같은 질문에 사용하세요."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": (
                            "분석할 종목의 이름(예: 'LG전자') 또는 "
                            "6자리 종목코드(예: '066570')"
                        ),
                    }
                },
                "required": ["identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_price",
            "description": (
                "특정 날짜의 종목 종가를 조회합니다. "
                "'삼성전자 2026년 5월 18일 주가', '작년 12월 31일 카카오 주가' 같은 질문에 사용하세요. "
                "오늘 현재주가는 get_naver_finance를 사용하세요."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "종목명(예: '삼성전자') 또는 6자리 종목코드(예: '005930')",
                    },
                    "date": {
                        "type": "string",
                        "description": "조회할 날짜. 'YYYY-MM-DD', 'YY년 M월 D일', 'YYYY.MM.DD' 형식 모두 가능",
                    },
                },
                "required": ["identifier", "date"],
            },
        },
    },
]


def _build_system_prompt() -> str:
    # 실제 계좌 잔고 (KIS API 우선, 실패 시 인메모리 positions 폴백)
    cash_text = "  (현금 정보 없음)"
    positions_text = "  현재 보유 종목 없음"
    try:
        from trader import KISTrader
        from config import KIS_APP_KEY
        if KIS_APP_KEY:
            t = KISTrader()
            holdings = t.get_balance()
            cash = t.get_available_cash()
            cash_text = f"  주문 가능 현금: {cash:,}원"
            if holdings:
                pos_lines = []
                for h in holdings:
                    pos_lines.append(
                        f"  - {h['name']} ({h['stock_code']}): "
                        f"평균매수가 {h['avg_price']:,}원, "
                        f"수량 {h['qty']}주, "
                        f"평가손익 {h['eval_profit']:+,}원"
                    )
                positions_text = "\n".join(pos_lines)
        else:
            raise ValueError("KIS_APP_KEY 없음")
    except Exception:
        # KIS API 실패 시 인메모리 positions 폴백
        try:
            from trader import positions
            if positions:
                pos_lines = []
                for ticker, info in positions.items():
                    name = STOCKS.get(ticker, ticker)
                    pos_lines.append(
                        f"  - {name} ({ticker}): "
                        f"평균매수가 {info.get('avg_price', 0):,.0f}원, "
                        f"수량 {info.get('quantity', 0)}주"
                    )
                positions_text = "\n".join(pos_lines)
        except Exception:
            positions_text = "  (포지션 정보 로드 실패)"

    stocks_text = "\n".join(
        [f"  - {name} ({ticker})" for ticker, name in STOCKS.items()]
    ) or "  (종목 없음)"

    return f"""당신은 주식 자동매매 시스템의 AI 어시스턴트입니다.
사용자의 매매 전략·보유 현황을 정확히 알고 있으며, 주식 관련 질문에 도움을 줍니다.
답변은 반드시 한국어로 하세요.

━━━ 툴 사용 규칙 ━━━
- 실시간 재무 수치(PER, EPS 등) + 현재주가 한국 주식 → get_naver_finance
- 실시간 재무 수치 미국 주식 → get_yahoo_finance
- 특정 날짜의 과거 주가 조회 → get_historical_price
- 종목 신호/원칙 분석 → get_stock_signal
- 뉴스 검색 → get_naver_news
- 이미 알고 있는 정보(전략, 원칙, 파라미터)는 툴 없이 바로 답변
- 사용자가 매수/매도/전량매도를 명시적으로 요청하면 반드시 propose_trade 툴을 호출하세요.
  텍스트로만 확인을 묻지 말고 반드시 툴을 호출해야 합니다.
- "현재주가 기준 PER/PBR 계산" 요청 시: get_naver_finance 호출 후 반환된 현재주가와
  EPS/BPS 수치로 직접 계산하세요 (PER = 현재주가 ÷ EPS, PBR = 현재주가 ÷ BPS).
  네이버 사전계산 PER/PBR을 그대로 반환하지 마세요.
- 특정 연도 예상EPS 기반 계산 시: 연도가 명시된 항목(예: EPS(2026/12(E)))을 사용하세요.

━━━ 매매 전략 원칙 ━━━

▶ 매수 1원칙 (시가돌파 — 분봉 기반, 9:30 이후 장중 실시간)
  - 전일 종가 > MA{MA_SHORT} (5일선 위에 있는 상태)
  - 9:00~9:30 사이 저가가 장 시작 시가(9:00 첫 캔들 open) 아래로 한 번이라도 내려간 적 있을 것
  - 9:30 이후 현재가가 장 시작 시가를 상향 돌파 (직전 캔들 ≤ 시가 < 현재 캔들)
  - 돌파 캔들 거래량 > 직전 5분봉 평균 거래량 × {VOLUME_INCREASE_RATIO}배

▶ 매수 2원칙 (MA사이반등 — 일봉 기반)
  - 전일 종가가 MA{MA_SHORT}(5일선)~MA{MA_LONG}(20일선) 사이
  - 당일 거래량 증가 (50일 평균 × {VOLUME_INCREASE_RATIO}배 이상) + 양봉

▶ 매수 3원칙 (MA20아래급등 — 일봉 기반)
  - 전일 종가 < MA{MA_LONG} (20일선 아래)
  - 당일 거래량 급증 (50일 평균 × {VOLUME_SURGE_RATIO}배 이상) + 양봉 또는 도지

▶ 매도 1원칙 (부분매도 50%)
  - 종가 > MA{MA_SHORT} 상태에서 장대음봉 + 거래량 급증 (50일 평균 × {VOLUME_SURGE_RATIO}배)
  - 실행: 보유량 50% 시장가 매도

▶ 매도 2원칙 (전량매도)
  - 종가가 MA{MA_SHORT}~MA{MA_LONG} 사이 + 거래량 증가 (× {VOLUME_INCREASE_RATIO}) + 음봉
  - 실행: 전량 시장가 매도

━━━ 거래량 파라미터 ━━━
  평균 산정 기간 : 직전 {VOLUME_LOOKBACK_DAYS}일
  증가 기준      : 50일 평균 × {VOLUME_INCREASE_RATIO}배
  급증 기준(일봉): 50일 평균 × {VOLUME_SURGE_RATIO}배
  급증 기준(분봉): 직전 5분봉 평균 × {VOLUME_SURGE_MINUTE_RATIO}배
  분봉 급증 감지 시 네이버 뉴스 3건 텔레그램 전송

━━━ 이동평균 ━━━
  단기: MA{MA_SHORT} (5일 단순이동평균)
  장기: MA{MA_LONG} (20일 단순이동평균)
  RSI 보조지표: 14일

━━━ 거래 실행 방식 ━━━
  매수 사이클 : 5분 간격 (9:05, 9:10 … 15:25)
  포지션 크기 : 총 자산(현금+평가액)의 40%로 시장가 매수
  매도 우선순위: 매도신호 먼저 체크 후 매수신호 체크

━━━ 계좌 현황 ━━━
{cash_text}

━━━ 현재 보유 종목 ━━━
{positions_text}

━━━ 매매 대상 종목 목록 ━━━
{stocks_text}"""


def ask(user_id: int, question: str) -> str:
    """
    사용자 질문에 GPT로 답변.
    필요 시 Naver Finance 툴을 자동 호출해 재무 데이터를 보강.
    """
    # ── 대기 중인 주문 확인 응답 처리 ──
    if user_id in _pending_orders:
        lower = question.strip().lower()
        if any(w in lower for w in _CONFIRM_WORDS):
            order  = _pending_orders.pop(user_id)
            result = _execute_trade_order(order)
            _histories.setdefault(user_id, []).append({"role": "assistant", "content": result})
            return result
        elif any(w in lower for w in _DENY_WORDS):
            _pending_orders.pop(user_id)
            msg = "🚫 주문을 취소했습니다."
            _histories.setdefault(user_id, []).append({"role": "assistant", "content": msg})
            return msg

    history = _histories.setdefault(user_id, [])
    history.append({"role": "user", "content": question})

    # 히스토리 크기 제한
    if len(history) > _MAX_HISTORY:
        _histories[user_id] = history[-_MAX_HISTORY:]
        history = _histories[user_id]

    messages = [{"role": "system", "content": _build_system_prompt()}] + history

    try:
        resp = client.chat.completions.create(
            model="gpt-4.5",
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # ── 툴 호출이 있으면 처리 ──
        tool_results: list[str] = []
        if msg.tool_calls:
            messages.append(msg)

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                identifier = args.get("identifier", "")
                if tc.function.name == "set_conditional_order":
                    result = _call_set_conditional_order(
                        args.get("stock_name", ""),
                        args.get("condition_type", ""),
                        float(args.get("condition_value", 0)),
                        args.get("action", ""),
                        int(args.get("quantity", 0)),
                    )
                    logger.info("[GPT] 조건부 주문 등록: %s", args)
                elif tc.function.name == "list_conditional_orders":
                    result = _call_list_conditional_orders()
                elif tc.function.name == "cancel_conditional_order":
                    result = _call_cancel_conditional_order(args.get("order_id", ""))
                elif tc.function.name == "propose_trade":
                    act  = args.get("action", "")
                    code = args.get("stock_code", "")
                    name = args.get("stock_name", "")
                    qty  = int(args.get("quantity", 0))
                    _pending_orders[user_id] = {
                        "action": act, "stock_code": code,
                        "stock_name": name, "quantity": qty,
                    }
                    action_label = {"buy": "매수", "sell": "매도", "sellall": "전량 매도"}.get(act, act)
                    qty_str = f"{qty}주 " if act != "sellall" else ""
                    result = (
                        f"⚠️ <b>{name}({code})</b> {qty_str}시장가 {action_label}\n"
                        f"확인하시겠습니까? (예 / 아니오)"
                    )
                    logger.info("[GPT] 주문 제안: %s %s %s%s", act, code, qty_str, action_label)
                elif tc.function.name == "get_yahoo_finance":
                    yt = args.get("ticker", "")
                    logger.info("[GPT] Yahoo Finance 조회 요청: %s", yt)
                    result = _call_yahoo_finance(yt)
                elif tc.function.name == "get_naver_finance":
                    logger.info("[GPT] Naver Finance 조회 요청: %s", identifier)
                    result = _call_naver_finance(identifier)
                elif tc.function.name == "get_historical_price":
                    date_arg = args.get("date", "")
                    logger.info("[GPT] 과거 주가 조회 요청: %s %s", identifier, date_arg)
                    result = _call_historical_price(identifier, date_arg)
                elif tc.function.name == "get_stock_signal":
                    logger.info("[GPT] 종목 신호 분석 요청: %s", identifier)
                    result = _call_stock_signal(identifier)
                elif tc.function.name == "get_naver_news":
                    query = args.get("query", "")
                    n     = int(args.get("n", 5))
                    logger.info("[GPT] 네이버 뉴스 검색 요청: %s (%d건)", query, n)
                    result = _call_naver_news(query, n)
                else:
                    result = f"알 수 없는 툴: {tc.function.name}"
                tool_results.append(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # 툴 결과 포함해서 최종 답변 생성
            resp2 = client.chat.completions.create(
                model="gpt-4.5",
                messages=messages,
            )
            answer = resp2.choices[0].message.content or ""

        else:
            answer = msg.content or ""

    except Exception as e:
        logger.error("GPT 응답 오류: %s", e)
        answer = f"⚠️ GPT 응답 오류: {e}"

    # ── 툴을 사용한 경우 Context Recall 평가 (서브에이전트) ──
    if tool_results:
        score = _eval_context_recall("\n\n".join(tool_results), answer)
        answer_with_score = f"{answer}\n\n─────────────────\n📊 Context Recall: {score:.2f}"
    else:
        answer_with_score = answer

    history.append({"role": "assistant", "content": answer})  # 히스토리엔 점수 제외
    return answer_with_score


def _eval_context_recall(context: str, answer: str) -> float:
    """
    서브에이전트 GPT가 Context Recall을 평가.
    context: 툴에서 검색된 실제 데이터
    answer:  메인 GPT가 생성한 최종 답변
    반환: 0.0 ~ 1.0
    """
    prompt = f"""당신은 RAG 평가 전문가입니다.
아래 [검색된 컨텍스트]와 [생성된 답변]을 분석하여 Context Recall 점수를 계산하세요.

Context Recall = (컨텍스트로 근거를 찾을 수 있는 답변 내 주장 수) / (답변 내 전체 주장 수)

[검색된 컨텍스트]
{context}

[생성된 답변]
{answer}

0.00~1.00 사이의 숫자만 반환하세요. 설명 없이 숫자만."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4.5-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        return round(min(max(float(raw), 0.0), 1.0), 2)
    except Exception as e:
        logger.warning("Context Recall 평가 실패: %s", e)
        return -1.0


def clear_history(user_id: int) -> None:
    """유저의 대화 기록 초기화."""
    _histories.pop(user_id, None)


def _call_set_conditional_order(stock_name: str, condition_type: str,
                                condition_value: float, action: str, quantity: int) -> str:
    try:
        from conditional_orders import add_order
        # STOCKS에서 티커/코드 탐색
        ticker, stock_code = "", ""
        for t, name in STOCKS.items():
            if name == stock_name:
                ticker = t
                stock_code = t.replace(".KS", "").replace(".KQ", "")
                break
        if not ticker:
            return f"'{stock_name}' 종목을 STOCKS 목록에서 찾을 수 없습니다."

        order = add_order(ticker, stock_name, stock_code,
                          condition_type, condition_value, action, quantity)

        ctype_label = {
            "price_below":  f"현재가 < {condition_value:,.0f}원",
            "price_above":  f"현재가 > {condition_value:,.0f}원",
            "profit_above": f"수익률 > +{condition_value}%",
            "profit_below": f"수익률 < {condition_value}%",
        }.get(condition_type, condition_type)
        action_label = {"buy": "매수", "sell": "매도", "sellall": "전량 매도"}.get(action, action)
        qty_str = f"{quantity}주 " if action != "sellall" else ""

        return (
            f"✅ 조건부 주문 등록 완료 [#{order['id']}]\n"
            f"  종목: {stock_name}({stock_code})\n"
            f"  조건: {ctype_label}\n"
            f"  실행: {action_label} {qty_str}(시장가)\n"
            f"  5분 간격으로 조건 체크 후 자동 실행됩니다."
        )
    except Exception as e:
        logger.error("조건부 주문 등록 오류: %s", e)
        return f"조건부 주문 등록 실패: {e}"


def _call_list_conditional_orders() -> str:
    try:
        from conditional_orders import list_orders
        orders = list_orders()
        if not orders:
            return "등록된 조건부 주문이 없습니다."
        lines = [f"[조건부 주문 목록 — {len(orders)}건]"]
        for o in orders:
            ctype_label = {
                "price_below":  f"현재가 < {float(o['condition_value']):,.0f}원",
                "price_above":  f"현재가 > {float(o['condition_value']):,.0f}원",
                "profit_above": f"수익률 > +{o['condition_value']}%",
                "profit_below": f"수익률 < {o['condition_value']}%",
            }.get(o["condition_type"], o["condition_type"])
            action_label = {"buy": "매수", "sell": "매도", "sellall": "전량 매도"}.get(o["action"], o["action"])
            qty_str = f"{o['quantity']}주 " if o["action"] != "sellall" else ""
            lines.append(
                f"  #{o['id']} | {o['stock_name']}({o['stock_code']}) | "
                f"조건: {ctype_label} → {action_label} {qty_str}| 등록: {o['created_at']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"조건부 주문 조회 실패: {e}"


def _call_cancel_conditional_order(order_id: str) -> str:
    try:
        from conditional_orders import cancel_order, list_orders
        if order_id.lower() == "all":
            orders = list_orders()
            for o in orders:
                cancel_order(o["id"])
            return f"✅ 조건부 주문 {len(orders)}건 전체 취소 완료"
        if cancel_order(order_id):
            return f"✅ 조건부 주문 #{order_id} 취소 완료"
        return f"⚠️ #{order_id}를 찾을 수 없습니다. list_conditional_orders로 ID를 확인하세요."
    except Exception as e:
        return f"조건부 주문 취소 실패: {e}"


def _execute_trade_order(order: dict) -> str:
    """확인된 주문을 KIS API로 실제 실행."""
    from config import KIS_APP_KEY
    action     = order["action"]
    stock_code = order["stock_code"]
    stock_name = order["stock_name"]
    quantity   = order.get("quantity", 0)

    if not KIS_APP_KEY:
        return "⚠️ KIS_APP_KEY 미설정 — 시뮬레이션 모드에서는 실제 주문이 실행되지 않습니다."

    try:
        from trader import KISTrader
        t = KISTrader()
        if action == "buy":
            t.buy(stock_code, quantity)
            return f"✅ {stock_name}({stock_code}) {quantity}주 시장가 매수 주문 완료!"
        elif action == "sell":
            t.sell(stock_code, quantity)
            return f"✅ {stock_name}({stock_code}) {quantity}주 시장가 매도 주문 완료!"
        elif action == "sellall":
            balance = t.get_balance()
            holding = next((b for b in balance if b["stock_code"] == stock_code), None)
            if not holding or holding["qty"] == 0:
                return f"⚠️ {stock_name}({stock_code}) 보유 수량이 없습니다."
            t.sell(stock_code, holding["qty"])
            return f"✅ {stock_name}({stock_code}) {holding['qty']}주 전량 시장가 매도 주문 완료!"
        else:
            return f"⚠️ 알 수 없는 주문 유형: {action}"
    except Exception as e:
        logger.error("주문 실행 오류: %s", e)
        return f"⚠️ 주문 실패: {e}"


def _call_yahoo_finance(ticker: str) -> str:
    try:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())
        info = t.info

        if not info or info.get("quoteType") is None:
            return f"'{ticker}' 종목을 Yahoo Finance에서 찾을 수 없습니다."

        name    = info.get("longName") or info.get("shortName") or ticker
        currency = info.get("currency", "USD")

        def fmt_num(v, unit="", pct=False):
            if v is None:
                return "N/A"
            if pct:
                return f"{v * 100:.2f}%"
            if abs(v) >= 1e12:
                return f"{v / 1e12:.2f}T {currency}{unit}"
            if abs(v) >= 1e9:
                return f"{v / 1e9:.2f}B {currency}{unit}"
            if abs(v) >= 1e6:
                return f"{v / 1e6:.2f}M {currency}{unit}"
            return f"{v:,.2f}{unit}"

        lines = [f"[{name} ({ticker.upper()}) — Yahoo Finance 재무지표]", ""]

        # 주가 정보
        lines.append("● 주가")
        lines.append(f"  현재가      : {fmt_num(info.get('currentPrice'))} {currency}")
        lines.append(f"  52주 고/저  : {fmt_num(info.get('fiftyTwoWeekHigh'))} / {fmt_num(info.get('fiftyTwoWeekLow'))}")
        lines.append(f"  시가총액    : {fmt_num(info.get('marketCap'))}")

        # 밸류에이션
        lines.append("\n● 밸류에이션")
        lines.append(f"  PER (TTM)   : {fmt_num(info.get('trailingPE'))}")
        lines.append(f"  PER (FWD)   : {fmt_num(info.get('forwardPE'))}")
        lines.append(f"  PBR         : {fmt_num(info.get('priceToBook'))}")
        lines.append(f"  EPS (TTM)   : {fmt_num(info.get('trailingEps'))} {currency}")
        lines.append(f"  EPS (FWD)   : {fmt_num(info.get('forwardEps'))} {currency}")

        # 수익성
        lines.append("\n● 수익성")
        lines.append(f"  매출액      : {fmt_num(info.get('totalRevenue'))}")
        lines.append(f"  영업이익률  : {fmt_num(info.get('operatingMargins'), pct=True)}")
        lines.append(f"  순이익률    : {fmt_num(info.get('profitMargins'), pct=True)}")
        lines.append(f"  ROE         : {fmt_num(info.get('returnOnEquity'), pct=True)}")
        lines.append(f"  ROA         : {fmt_num(info.get('returnOnAssets'), pct=True)}")

        # 재무 건전성
        lines.append("\n● 재무 건전성")
        lines.append(f"  부채비율(D/E): {fmt_num(info.get('debtToEquity'))}")
        lines.append(f"  유동비율    : {fmt_num(info.get('currentRatio'))}")
        lines.append(f"  잉여현금흐름: {fmt_num(info.get('freeCashflow'))}")

        # 배당
        div_yield = info.get("dividendYield")
        lines.append("\n● 배당")
        lines.append(f"  시가배당률  : {fmt_num(div_yield, pct=True)}")
        lines.append(f"  주당배당금  : {fmt_num(info.get('dividendRate'))} {currency}")

        # 연간 실적 시계열 (최근 3년)
        try:
            fin = t.financials
            if fin is not None and not fin.empty:
                lines.append("\n● 연간 실적 (최근 3년)")
                cols = fin.columns[:3]
                for row_key, label in [
                    ("Total Revenue",    "매출액"),
                    ("Operating Income", "영업이익"),
                    ("Net Income",       "순이익"),
                ]:
                    if row_key in fin.index:
                        vals = "  |  ".join(
                            f"{str(c)[:4]}: {fmt_num(fin.loc[row_key, c])}"
                            for c in cols
                        )
                        lines.append(f"  {label}: {vals}")
        except Exception:
            pass

        return "\n".join(lines)

    except Exception as e:
        logger.error("Yahoo Finance 툴 오류: %s", e)
        return f"Yahoo Finance 조회 오류: {e}"


def _call_naver_news(query: str, n: int = 5) -> str:
    try:
        from news_fetcher import fetch_naver_news
        n = min(max(1, n), 10)
        items = fetch_naver_news(query, n=n)
        if not items:
            return f"'{query}' 관련 뉴스를 찾을 수 없습니다. (NAVER API 키 확인 필요)"
        lines = [f"[{query} — 네이버 최신 뉴스 {len(items)}건]"]
        for i, item in enumerate(items, 1):
            title   = item.get("title", "제목 없음")
            link    = item.get("link", "")
            pub     = item.get("pubDate", "")
            desc    = item.get("description", "")
            lines.append(f"\n{i}. {title}")
            if pub:
                lines.append(f"   📅 {pub}")
            if desc:
                lines.append(f"   {desc[:100]}{'...' if len(desc) > 100 else ''}")
            if link:
                lines.append(f"   🔗 {link}")
        return "\n".join(lines)
    except Exception as e:
        logger.error("네이버 뉴스 툴 오류: %s", e)
        return f"네이버 뉴스 조회 오류: {e}"


def _call_naver_finance(identifier: str) -> str:
    try:
        from naver_finance import get_financials
        return get_financials(identifier)
    except Exception as e:
        logger.error("Naver Finance 툴 오류: %s", e)
        return f"Naver Finance 조회 오류: {e}"


def _parse_korean_date(date_str: str):
    """한국어/일반 날짜 문자열 → datetime.date. 실패 시 None."""
    import re
    from datetime import date
    s = date_str.strip()
    for pattern, fmt in [
        (r"(\d{4})-(\d{1,2})-(\d{1,2})", lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
        (r"(\d{4})\.(\d{1,2})\.(\d{1,2})", lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
        (r"(\d{2,4})년\s*(\d{1,2})월\s*(\d{1,2})일", lambda m: date(int(m[1]) if int(m[1]) > 100 else 2000 + int(m[1]), int(m[2]), int(m[3]))),
        (r"(\d{8})", lambda m: date(int(m[1][:4]), int(m[1][4:6]), int(m[1][6:8]))),
    ]:
        match = re.search(pattern, s)
        if match:
            try:
                return fmt(match)
            except ValueError:
                continue
    return None


def _call_historical_price(identifier: str, date: str) -> str:
    try:
        import yfinance as yf
        import pandas as pd
        from datetime import timedelta
        from naver_finance import _resolve_code

        code, name = _resolve_code(identifier)
        if not code:
            return f"'{identifier}' 종목을 찾을 수 없습니다."

        target = _parse_korean_date(date)
        if not target:
            return f"날짜 형식을 인식할 수 없습니다: {date}\n'YYYY-MM-DD' 또는 'YY년 M월 D일' 형식으로 입력하세요."

        label = name or code
        start = (target - timedelta(days=7)).strftime("%Y-%m-%d")
        end   = (target + timedelta(days=7)).strftime("%Y-%m-%d")

        ticker = f"{code}.KS"
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            ticker = f"{code}.KQ"
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            return f"{label} {date} 주가 데이터를 가져올 수 없습니다."

        df.index = pd.to_datetime(df.index).normalize()
        target_ts = pd.Timestamp(target)

        if target_ts in df.index:
            close = float(df.loc[target_ts, "Close"].iloc[0] if hasattr(df.loc[target_ts, "Close"], "iloc") else df.loc[target_ts, "Close"])
            return f"{label} {target.strftime('%Y-%m-%d')} 종가: {close:,.0f}원"

        past = df[df.index <= target_ts]
        row_ts = past.index[-1] if not past.empty else df.index[0]
        close = float(df.loc[row_ts, "Close"].iloc[0] if hasattr(df.loc[row_ts, "Close"], "iloc") else df.loc[row_ts, "Close"])
        return (
            f"{label} {row_ts.strftime('%Y-%m-%d')} 종가: {close:,.0f}원"
            f" (요청일 {target.strftime('%Y-%m-%d')} 기준 가장 가까운 거래일)"
        )
    except Exception as e:
        logger.error("과거 주가 조회 오류: %s", e)
        return f"과거 주가 조회 오류: {e}"


def _call_stock_signal(identifier: str) -> str:
    try:
        from config import (
            STOCKS, MA_SHORT, MA_LONG, RSI_PERIOD,
            VOLUME_LOOKBACK_DAYS, VOLUME_INCREASE_RATIO, VOLUME_SURGE_RATIO,
        )
        from data_fetcher import fetch_ohlcv
        from indicators import add_all_indicators, detect_crossover
        from strategy import generate_signals, get_latest_signal

        # 티커 탐색
        ticker = None
        stock_name = identifier
        clean = identifier.replace(".KS", "").replace(".KQ", "").strip()

        for t, name in STOCKS.items():
            code = t.replace(".KS", "").replace(".KQ", "")
            if identifier.strip() in (t, code, name) or clean in (code, name):
                ticker = t
                stock_name = name
                break

        if not ticker:
            if clean.isdigit() and len(clean) == 6:
                import yfinance as yf
                for suffix in [".KS", ".KQ"]:
                    df_test = yf.download(f"{clean}{suffix}", period="5d", progress=False, auto_adjust=True)
                    if not df_test.empty:
                        ticker = f"{clean}{suffix}"
                        stock_name = clean
                        break
            if not ticker:
                return f"'{identifier}' 종목을 찾을 수 없습니다."

        stock_code = ticker.replace(".KS", "").replace(".KQ", "")

        # 데이터 수집 및 지표 계산
        df = fetch_ohlcv(ticker, period_years=1)
        try:
            from runner import _append_today_bar
            from data_fetcher import get_minute_data
            minute_df = None
            try:
                minute_df = get_minute_data(ticker, interval_min=1)
            except Exception:
                pass
            df = _append_today_bar(df, minute_df)
        except Exception:
            pass

        df = add_all_indicators(df, short=MA_SHORT, long=MA_LONG, rsi_period=RSI_PERIOD)
        df = detect_crossover(df, short=MA_SHORT, long=MA_LONG)
        df = generate_signals(df)
        sig = get_latest_signal(df)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        close      = sig["close"]
        ma5        = sig["ma_short"]
        ma20       = sig["ma_long"]
        rsi        = sig["rsi"]
        volume     = sig["volume"]
        open_price = float(last["Open"])
        prev_close = float(prev["Close"])
        prev_ma5   = float(prev[f"MA_{MA_SHORT}"])
        prev_ma20  = float(prev[f"MA_{MA_LONG}"])
        ma_min     = min(prev_ma5, prev_ma20)
        ma_max     = max(prev_ma5, prev_ma20)

        vol_avg   = float(df["Volume"].rolling(window=VOLUME_LOOKBACK_DAYS, min_periods=1).mean().shift(1).iloc[-1])
        vol_ratio = volume / vol_avg if vol_avg > 0 else 0

        candle = "양봉" if close > open_price else ("음봉" if close < open_price else "도지")

        def chk(cond): return "✅" if cond else "❌"

        c1_a = prev_close > prev_ma5
        c1_b = float(last["Low"]) < open_price
        c1_c = close > open_price
        c1   = bool(last.get("buy_signal_1", False))

        c2_a = ma_min < prev_close < ma_max
        c2_b = vol_ratio >= VOLUME_INCREASE_RATIO
        c2_c = close > open_price
        c2   = bool(last.get("buy_signal_2", False))

        c3_a = prev_close < prev_ma20
        c3_b = vol_ratio >= VOLUME_SURGE_RATIO
        c3_c = close >= open_price
        c3   = bool(last.get("buy_signal_3", False))

        lines = [
            f"[{stock_name} ({stock_code}) — {sig['date']} 기준 신호 분석]",
            f"",
            f"● 현재가: {close:,.0f}원  MA5: {ma5:,.0f}  MA20: {ma20:,.0f}  RSI: {rsi}",
            f"● 캔들: {candle} (시가 {open_price:,.0f}원)",
            f"● 거래량: {volume:,}주  ({vol_ratio:.2f}배 / {VOLUME_LOOKBACK_DAYS}일 평균 {vol_avg:,.0f}주)",
            f"",
            f"▶ 1원칙 (시가돌파)",
            f"  {chk(c1_a)} 전일 종가({prev_close:,.0f}) > MA5({prev_ma5:,.0f})",
            f"  {chk(c1_b)} 당일 저가({float(last['Low']):,.0f}) < 시가({open_price:,.0f})",
            f"  {chk(c1_c)} 당일 종가 > 시가 (양봉)",
            f"  → {'✅ 신호 발생' if c1 else '❌ 미발생'}",
            f"",
            f"▶ 2원칙 (MA사이반등)",
            f"  {chk(c2_a)} 전일 종가({prev_close:,.0f})가 MA5({prev_ma5:,.0f})~MA20({prev_ma20:,.0f}) 사이",
            f"  {chk(c2_b)} 거래량 비율 {vol_ratio:.2f}배 ≥ {VOLUME_INCREASE_RATIO}배 기준",
            f"  {chk(c2_c)} 양봉",
            f"  → {'✅ 신호 발생' if c2 else '❌ 미발생'}",
            f"",
            f"▶ 3원칙 (MA20아래급등)",
            f"  {chk(c3_a)} 전일 종가({prev_close:,.0f}) < MA20({prev_ma20:,.0f})",
            f"  {chk(c3_b)} 거래량 비율 {vol_ratio:.2f}배 ≥ {VOLUME_SURGE_RATIO}배 기준",
            f"  {chk(c3_c)} 양봉 또는 도지",
            f"  → {'✅ 신호 발생' if c3 else '❌ 미발생'}",
            f"",
            f"━━━ 종합 신호 ━━━",
        ]

        if sig["buy"]:
            lines.append(f"✅ 매수: {'/'.join(sig['buy_which'])}")
        elif sig["sell_partial"]:
            lines.append(f"🟡 매도(부분): {'/'.join(sig['sell_which'])}")
        elif sig["sell_full"]:
            lines.append(f"🔴 매도(전량): {'/'.join(sig['sell_which'])}")
        else:
            lines.append("⚪ 매수/매도 신호 없음")

        return "\n".join(lines)

    except Exception as e:
        logger.error("종목 신호 분석 오류: %s", e)
        return f"신호 분석 오류: {e}"
