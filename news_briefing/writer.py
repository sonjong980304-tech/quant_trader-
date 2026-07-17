"""
news_briefing/writer.py — closed-book 브리핑 작성
(플랜: .omc/plans/news-briefing-revamp-plan.md §3·§5 6단계)

상위 LLM 1콜로 closed-book 브리핑 초안(BriefingDraft)을 작성한다.
- write_briefing: 기사·시장 스냅샷·보유종목 뉴스만 근거로 4개 섹션 + KOSPI/KOSDAQ 전망 작성
- find_uncited: article_ids가 비어있는 사실 문장 idx 검출 (market_data 규약 반영)
- rewrite: 게이트 실패 피드백을 반영해 지적된 문장만 고친 새 draft 반환
- parse_draft: LLM 응답 파싱 + 스키마 검증 (단독 호출 가능)
"""
import json
import logging
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# BriefingDraft 스키마 화이트리스트
SECTIONS = ("nasdaq", "cause", "kr_impact", "holdings")
DIRECTIONS = ("up", "flat", "down")
MARKETS = ("KOSPI", "KOSDAQ")
MARKET_DATA_TAG = "market_data"

LlmCall = Callable[[str, str], str]

_SYSTEM_PROMPT = """당신은 국내 투자자를 위한 증시 브리핑 작성자입니다.

[closed-book 규칙 — 반드시 준수]
제공된 기사와 시장 데이터만 근거로 작성. 모든 사실 문장에 근거 기사 id를 article_ids로 명시.
기사에 없는 내용 서술 금지. 지수 수치는 제공된 시장 데이터의 값을 그대로 사용.

지수 수치만 서술하고 특정 기사에 근거하지 않는 문장은 article_ids에 기사 id 대신
문자열 "market_data" 하나만 담으세요 (article_ids를 빈 배열로 두는 것은 허용되지 않습니다).

[출력 구조]
- sentences: 아래 4개 섹션을 모두 포함하는 문장 배열 (idx는 0부터 연속된 정수)
  - nasdaq: 나스닥·S&P500·다우존스(dow)·필라델피아 반도체지수(sox, SOX) 등 미국 증시 마감 시황.
    market_snapshot에 있는 지수는 빠짐없이 언급하세요.
  - cause: 미국 증시 등락의 원인 분석
  - kr_impact: 미국 증시가 국내 증시에 미치는 영향
  - holdings: 보유종목 관련 뉴스
- forecasts: KOSPI, KOSDAQ 각각 정확히 1건씩, direction(up/flat/down)과 rationale 포함

반드시 아래 JSON 스키마로만 응답하세요. 다른 설명 텍스트를 덧붙이지 마세요.
{
  "sentences": [
    {"idx": 0, "section": "nasdaq", "text": "...", "article_ids": [1, 2]}
  ],
  "forecasts": [
    {"market": "KOSPI", "direction": "up", "rationale": "..."},
    {"market": "KOSDAQ", "direction": "flat", "rationale": "..."}
  ]
}
"""

_RETRY_NOTICE = (
    "\n\n[재요청] 이전 응답이 JSON 스키마를 따르지 않았습니다. "
    "설명 없이 지정된 JSON 스키마로만 다시 응답하세요."
)


def _default_llm_call(system: str, user: str) -> str:
    """실제 OpenAI 호출 (llm_call 미주입 시 기본 경로). 테스트는 항상 llm_call을 주입해 우회한다."""
    import config
    from openai import OpenAI

    client = OpenAI(api_key=getattr(config, "OPENAI_API_KEY", ""))
    model = getattr(config, "NEWS_LLM_PREMIUM", "gpt-4-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def _extract_json(raw: str) -> dict:
    """LLM 응답에서 JSON 객체를 추출. 코드펜스 등 잡텍스트가 섞여도 첫 {부터 마지막 }까지 파싱 시도."""
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("응답에서 JSON 객체를 찾을 수 없습니다")
    return json.loads(raw[start:end + 1])


def parse_draft(raw: str) -> dict:
    """BriefingDraft 스키마 검증 포함 파싱. 스키마 위반 시 ValueError."""
    data = _extract_json(raw)

    if not isinstance(data, dict):
        raise ValueError("최상위 응답은 JSON 객체여야 합니다")

    sentences = data.get("sentences")
    forecasts = data.get("forecasts")
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("sentences가 비어있거나 배열이 아닙니다")
    if not isinstance(forecasts, list) or not forecasts:
        raise ValueError("forecasts가 비어있거나 배열이 아닙니다")

    for expected_idx, s in enumerate(sentences):
        if not isinstance(s, dict):
            raise ValueError(f"sentences[{expected_idx}]가 객체가 아닙니다")
        if s.get("idx") != expected_idx:
            raise ValueError(f"idx 연속성 위반: {expected_idx}번째 문장의 idx={s.get('idx')!r}")
        if s.get("section") not in SECTIONS:
            raise ValueError(f"알 수 없는 section: {s.get('section')!r}")
        if not isinstance(s.get("text"), str) or not s["text"].strip():
            raise ValueError(f"sentences[{expected_idx}].text가 비어있습니다")
        if not isinstance(s.get("article_ids"), list):
            raise ValueError(f"sentences[{expected_idx}].article_ids가 배열이 아닙니다")

    seen_markets = set()
    for f in forecasts:
        if not isinstance(f, dict):
            raise ValueError("forecasts 항목이 객체가 아닙니다")
        market = f.get("market")
        if market not in MARKETS:
            raise ValueError(f"알 수 없는 market: {market!r}")
        if f.get("direction") not in DIRECTIONS:
            raise ValueError(f"알 수 없는 direction: {f.get('direction')!r}")
        if not isinstance(f.get("rationale"), str) or not f["rationale"].strip():
            raise ValueError("forecasts.rationale이 비어있습니다")
        seen_markets.add(market)

    missing = set(MARKETS) - seen_markets
    if missing:
        raise ValueError(f"forecasts에 {missing} 시장이 누락됐습니다")

    return {"sentences": sentences, "forecasts": forecasts}


def _call_and_parse(system: str, user: str, llm_call: LlmCall) -> dict:
    """llm_call 호출 → parse_draft. 파싱 실패 시 1회 재요청, 재실패하면 ValueError."""
    last_err: Optional[Exception] = None
    for attempt in range(2):
        raw = llm_call(system, user)
        try:
            return parse_draft(raw)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            logger.warning("브리핑 draft 파싱 실패 (attempt=%d): %s", attempt + 1, e)
            user = user + _RETRY_NOTICE
    raise ValueError(f"LLM 응답 파싱 실패 (2회 시도): {last_err}")


def _format_articles(articles: List[dict]) -> str:
    if not articles:
        return "(제공된 기사 없음)"
    lines = []
    for a in articles:
        lines.append(
            f"[기사 {a.get('id')}] ({a.get('domain', '')}) {a.get('title', '')}\n"
            f"URL: {a.get('url', '')}\n"
            f"본문: {a.get('body', '')}"
        )
    return "\n\n".join(lines)


def write_briefing(
    articles: List[dict],
    market_snapshot: dict,
    stock_news: List[dict],
    llm_call: Optional[LlmCall] = None,
) -> dict:
    """상위 LLM 1콜로 closed-book 브리핑 초안(BriefingDraft)을 작성한다."""
    call = llm_call or _default_llm_call

    user_prompt = f"""=== 기사 목록 ===
{_format_articles(articles)}

=== 시장 스냅샷 ===
{json.dumps(market_snapshot, ensure_ascii=False, indent=2)}

=== 보유종목 선별 뉴스 ===
{json.dumps(stock_news, ensure_ascii=False, indent=2) if stock_news else "(보유종목 선별 뉴스 없음)"}

위 자료만 근거로 nasdaq/cause/kr_impact/holdings 4개 섹션의 문장과
KOSPI/KOSDAQ 전망을 지정된 JSON 스키마로 작성하세요."""

    return _call_and_parse(_SYSTEM_PROMPT, user_prompt, call)


def find_uncited(draft: dict) -> List[int]:
    """article_ids가 비어있는 사실 문장(4개 섹션 한정)의 idx 목록.
    market_data 태그(article_ids=["market_data"])가 담긴 문장은 인용된 것으로 간주한다."""
    uncited = []
    for s in draft.get("sentences", []):
        if s.get("section") not in SECTIONS:
            continue
        if not s.get("article_ids"):
            uncited.append(s.get("idx"))
    return uncited


def rewrite(
    draft: dict,
    articles: List[dict],
    market_snapshot: dict,
    feedback: List[dict],
    llm_call: Optional[LlmCall] = None,
) -> dict:
    """게이트 실패 피드백(위반 문장 idx·사유)을 반영해 해당 부분만 고친 새 draft를 반환한다 (1콜)."""
    call = llm_call or _default_llm_call

    feedback_lines = "\n".join(
        f"- idx {fb.get('idx')}: {fb.get('reason', '')}" for fb in (feedback or [])
    ) or "(피드백 없음)"

    user_prompt = f"""=== 기존 draft ===
{json.dumps(draft, ensure_ascii=False, indent=2)}

=== 기사 목록 ===
{_format_articles(articles)}

=== 시장 스냅샷 ===
{json.dumps(market_snapshot, ensure_ascii=False, indent=2)}

=== 게이트 실패 피드백 (아래 idx 문장만 수정할 것, 나머지는 그대로 유지) ===
{feedback_lines}

위 피드백을 반영해 지적된 문장만 closed-book 규칙에 맞게 다시 작성하고,
전체 draft를 지정된 JSON 스키마로 다시 출력하세요."""

    return _call_and_parse(_SYSTEM_PROMPT, user_prompt, call)
