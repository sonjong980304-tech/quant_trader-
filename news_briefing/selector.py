"""
news_briefing/selector.py - 저가 LLM 1콜 기사 선별

역할:
  1. 매크로(나스닥 시황·원인·한국영향) 관련 기사 선별
  2. 보유종목별 중요 뉴스 선별 + 3기준 카테고리 라벨링
     - event: 주가 영향 이벤트 (실적 발표, 신제품, 계약 체결 등)
     - cause: 급등락 원인 (수급·매크로·업황 등 가격 변동 배경 설명)
     - risk : 리스크 경보 (소송, 규제, 신용등급, 공급망 등 경계 신호)

입력/출력 데이터 계약은 .omc/plans/news-briefing-revamp-plan.md §3 selector.py 참고.
LLM 응답은 JSON을 강제하며, 파싱 실패 시 1회 재요청 후에도 실패하면 ValueError.
선별 총합(매크로+종목)은 12건을 넘지 않도록 LLM이 준 순서대로 절단한다.
"""

import json
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────

MAX_SELECTED = 12
CATEGORIES = ("event", "cause", "risk")
_BODY_PREVIEW_LEN = 200   # 본문 전체가 아닌 요약(미리보기)만 프롬프트에 전달 — 토큰 절약


# ─────────────────────────────────────────────
# 프롬프트 구성
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """당신은 금융 뉴스 선별 어시스턴트입니다. 아래 두 가지 작업을 한 번에 수행하세요.

[작업1: 매크로 기사 선별]
나스닥 시황, 미국 증시 등락 원인, 한국 시장에 미치는 영향과 관련된 기사를 선별하세요.

[작업2: 종목별 중요 뉴스 선별 + 카테고리 라벨링]
보유종목과 직접 관련된 중요 뉴스를 선별하고, 아래 3가지 기준 중 하나로 분류하세요.
- event: 주가에 영향을 줄 수 있는 이벤트 (실적 발표, 신제품, 계약 체결, 인수합병 등)
- cause: 해당 종목의 최근 급등락 원인을 설명하는 뉴스
- risk : 리스크 경보 (소송, 규제, 신용등급 하락, 공급망 이슈 등 경계가 필요한 신호)

반드시 아래 JSON 형식으로만 응답하세요. 다른 설명은 절대 추가하지 마세요.
{
  "macro_article_ids": [정수, ...],
  "stock_news": [
    {"ticker": "종목코드", "article_id": 정수, "category": "event|cause|risk", "summary": "한 줄 요약"}
  ]
}
"""


def _build_user_prompt(articles, holdings):
    """기사는 id·제목·요약(본문 미리보기)만 전달 — 본문 전체는 전달하지 않는다."""
    holding_lines = "\n".join(
        f"- {h.get('ticker', '')}: {h.get('name', '')}" for h in holdings
    )
    article_lines = []
    for a in articles:
        body = a.get("body") or ""
        preview = body[:_BODY_PREVIEW_LEN]
        article_lines.append(
            f"[id={a.get('id')}] ({a.get('domain', '')}) {a.get('title', '')}\n요약: {preview}"
        )
    articles_block = "\n\n".join(article_lines)

    return f"""[보유종목 목록]
{holding_lines}

[기사 목록]
{articles_block}
"""


# ─────────────────────────────────────────────
# 기본 LLM 호출 (OpenAI, NEWS_LLM_CHEAP)
# ─────────────────────────────────────────────

def _default_llm_call(system, user):
    from openai import OpenAI
    import config

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    model = getattr(config, "NEWS_LLM_CHEAP", "gpt-4-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


# ─────────────────────────────────────────────
# 응답 파싱 + 검증
# ─────────────────────────────────────────────

def _strip_code_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def _parse_response(raw):
    """JSON 파싱 + 데이터 계약 검증. 실패하면 None(재요청 트리거)."""
    if not raw:
        return None

    try:
        data = json.loads(_strip_code_fence(raw))
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    macro_ids = data.get("macro_article_ids")
    stock_news = data.get("stock_news")
    if not isinstance(macro_ids, list) or not isinstance(stock_news, list):
        return None

    try:
        macro_ids = [int(x) for x in macro_ids]
    except (TypeError, ValueError):
        return None

    cleaned_stock_news = []
    for item in stock_news:
        if not isinstance(item, dict):
            return None
        try:
            ticker = str(item["ticker"])
            article_id = int(item["article_id"])
            category = str(item["category"])
            summary = str(item["summary"])
        except (KeyError, TypeError, ValueError):
            return None
        if category not in CATEGORIES:
            logger.warning("selector: 알 수 없는 카테고리 '%s' — 제외", category)
            continue
        cleaned_stock_news.append({
            "ticker": ticker,
            "article_id": article_id,
            "category": category,
            "summary": summary,
        })

    return {"macro_article_ids": macro_ids, "stock_news": cleaned_stock_news}


def _truncate(result):
    """매크로+종목 총합이 12건을 넘으면 LLM이 준 순서대로(매크로 우선) 절단."""
    macro_ids = result["macro_article_ids"]
    stock_news = result["stock_news"]

    total = len(macro_ids) + len(stock_news)
    if total <= MAX_SELECTED:
        return result

    new_macro = macro_ids[:MAX_SELECTED]
    remaining = MAX_SELECTED - len(new_macro)
    new_stock = stock_news[:remaining] if remaining > 0 else []
    return {"macro_article_ids": new_macro, "stock_news": new_stock}


# ─────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────

def select_articles(articles, holdings, llm_call=None):
    """
    articles:  [{"id", "url", "domain", "source_lang", "title", "body", "published_at"}, ...]
    holdings:  [{"ticker", "name", "source"}, ...]
    llm_call:  (system: str, user: str) -> str  (미지정 시 OpenAI NEWS_LLM_CHEAP 사용)

    반환: {"macro_article_ids": [int, ...], "stock_news": [{"ticker","article_id","category","summary"}, ...]}
    파싱 실패가 재요청 1회 후에도 이어지면 ValueError.
    """
    if llm_call is None:
        llm_call = _default_llm_call

    user_prompt = _build_user_prompt(articles, holdings)

    raw = llm_call(_SYSTEM_PROMPT, user_prompt)
    result = _parse_response(raw)

    if result is None:
        logger.warning("selector: 응답 파싱 실패 — 1회 재요청")
        retry_prompt = (
            user_prompt
            + "\n\n(직전 응답이 올바른 JSON 형식이 아니었습니다. "
              "다른 설명 없이 지정된 JSON 객체만 다시 출력하세요.)"
        )
        raw = llm_call(_SYSTEM_PROMPT, retry_prompt)
        result = _parse_response(raw)

    if result is None:
        raise ValueError("selector: LLM 응답을 JSON으로 파싱하지 못했습니다 (재요청 포함 2회 실패)")

    return _truncate(result)
