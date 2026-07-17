"""
news_briefing/verifier.py — 브리핑 초안 사실·근거·링크 검증
(플랜: .omc/plans/news-briefing-revamp-plan.md §3·§5 7단계)

writer.py가 만든 BriefingDraft를 게이트(gate.py, US-008)에 넘기기 전에 3가지 축으로 검증한다.
- check_facts: 지수 등락률 클레임 vs market_snapshot 실측치 대조 (허용오차 기반, hard)
- check_grounding: 저가 LLM 배치 1콜로 문장별 근거(entailment) 판정 (soft — 게이트가 흡수)
- check_links: 기사 도메인 화이트리스트·HTTP 상태 체크 (hard)

violation dict 계약(gate.py가 그대로 소비): {"idx": int|None, "type": "fact"|"grounding"|"link", "reason": str}
(check_links는 idx를 알 수 없으므로 None + article_id를 추가로 담아 상위에서 문장 매핑에 사용)
"""
import json
import logging
import re
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

from news_briefing.writer import SECTIONS, MARKET_DATA_TAG  # noqa: E402  (순환 임포트 없음 — writer가 verifier를 참조하지 않음)

# market_data 태그 문장에서 지수 등락률 클레임을 추출하기 위한 별칭·방향어 사전
_MARKET_ALIASES = {
    "nasdaq": ("나스닥종합", "나스닥"),
    "sp500": ("S&P500", "S&P 500", "에스앤피500"),
    "dow": ("다우존스", "다우"),
    "sox": ("필라델피아 반도체지수", "필라델피아반도체지수", "반도체지수", "SOX"),
    "kospi": ("코스피", "KOSPI"),
    "kosdaq": ("코스닥", "KOSDAQ"),
}
_UP_WORDS = ("상승", "올랐", "오름", "강세")
_DOWN_WORDS = ("하락", "내렸", "내림", "떨어", "약세")


def extract_market_claims(draft: dict) -> List[dict]:
    """market_data 태그 문장에서 지수명 근처(25자 이내)의 %수치를 정규식으로 추출해
    check_facts에 넘길 claims([{"idx","market","claimed_pct"}])를 만든다.

    부호가 명시되지 않은 수치는 문장 내 방향어(상승/하락 등)로 부호를 보정한다.
    한 문장에 상승·하락이 둘 다 등장(예: "나스닥은 올랐고 다우는 내렸다")하면 어느 지수가
    어느 방향인지 정규식만으로는 안전하게 특정할 수 없으므로, 그 문장의 해당 지수는
    추출을 건너뛴다(오탐으로 올바른 문장을 게이트가 잘못 쳐내는 것보다 놓치는 편이 안전)."""
    claims = []
    for s in draft.get("sentences", []):
        if s.get("article_ids") != [MARKET_DATA_TAG]:
            continue
        text = s.get("text", "") or ""
        idx = s.get("idx")
        has_up = any(w in text for w in _UP_WORDS)
        has_down = any(w in text for w in _DOWN_WORDS)

        for market, aliases in _MARKET_ALIASES.items():
            match = None
            for alias in aliases:
                pattern = r"{}[^%\n]{{0,25}}?([+-]?\d+(?:[.,]\d+)?)\s*%".format(re.escape(alias))
                match = re.search(pattern, text)
                if match:
                    break
            if not match:
                continue

            raw = match.group(1).replace(",", ".")
            try:
                magnitude = abs(float(raw))
            except ValueError:
                continue

            if raw.startswith("+") or raw.startswith("-"):
                claimed_pct = float(raw)
            elif has_up and not has_down:
                claimed_pct = magnitude
            elif has_down and not has_up:
                claimed_pct = -magnitude
            else:
                continue  # 방향 모호 — 안전하게 스킵

            claims.append({"idx": idx, "market": market, "claimed_pct": claimed_pct})

    return claims

LlmCall = Callable[[str, str], str]

# 허용오차: 등락률(%) 비교는 ±0.05%p, 지수 레벨(종가) 비교는 ±0.1
_CHANGE_PCT_TOLERANCE = 0.05
_INDEX_LEVEL_TOLERANCE = 0.1

_GROUNDING_SYSTEM_PROMPT = """당신은 증시 브리핑 문장의 근거 여부를 판정하는 검증자입니다.

각 문장이 그 문장에 표시된 인용 기사(article_ids)의 내용으로 실제로 뒷받침되는지 판단하세요.
문장에 있으나 인용된 기사에서 확인할 수 없는 내용이 있으면 grounded를 false로 판정하세요.

반드시 아래 JSON 배열로만 응답하세요. 다른 설명 텍스트를 덧붙이지 마세요.
[
  {"idx": 1, "grounded": true, "reason": "..."},
  {"idx": 2, "grounded": false, "reason": "..."}
]
"""

_GROUNDING_RETRY_NOTICE = (
    "\n\n[재요청] 이전 응답이 JSON 배열 스키마를 따르지 않았습니다. "
    "설명 없이 지정된 JSON 배열로만 다시 응답하세요."
)


_FLOAT_EPSILON = 1e-9


def _within_tolerance(claimed_pct: float, actual_pct: float, is_index_level: bool = False) -> bool:
    """claimed_pct(클레임 값)와 actual_pct(실측치)의 차이가 허용오차 이내인지 판정한다.
    is_index_level=False: 등락률 비교(±0.05%p), True: 지수 레벨 비교(±0.1). 경계값은 통과로 처리한다.
    부동소수점 표현 오차(예: 1.55-1.50이 0.05보다 근소하게 커지는 경우) 때문에 경계값이
    오탐으로 실패하지 않도록 엡실론을 더해 비교한다."""
    tolerance = _INDEX_LEVEL_TOLERANCE if is_index_level else _CHANGE_PCT_TOLERANCE
    return abs(claimed_pct - actual_pct) <= tolerance + _FLOAT_EPSILON


def check_facts(claims: List[dict], market_snapshot: dict) -> List[dict]:
    """claims(수치 클레임 목록)를 market_snapshot의 실측 change_pct와 대조한다.
    claims: [{"idx": int, "market": "nasdaq"|"sp500"|"kospi"|"kosdaq", "claimed_pct": float}]
    실측 데이터가 없거나(None) market_snapshot에 해당 시장이 없으면 검증 불가로 보고 건너뛴다(예외 아님)."""
    violations = []
    for claim in claims or []:
        market = claim.get("market")
        snap = market_snapshot.get(market) if isinstance(market_snapshot, dict) else None
        if not isinstance(snap, dict):
            continue

        actual_pct = snap.get("change_pct")
        claimed_pct = claim.get("claimed_pct")
        if actual_pct is None or claimed_pct is None:
            continue

        if not _within_tolerance(claimed_pct, actual_pct, is_index_level=False):
            diff = abs(claimed_pct - actual_pct)
            violations.append({
                "idx": claim.get("idx"),
                "type": "fact",
                "reason": (
                    f"{market} 등락률 클레임 {claimed_pct}%가 실측 {actual_pct}%와 "
                    f"{diff:.4f}%p 차이로 허용오차(±{_CHANGE_PCT_TOLERANCE}%p)를 벗어남"
                ),
            })
    return violations


def _default_cheap_llm_call(system: str, user: str) -> str:
    """실제 저가 LLM 호출 (llm_call 미주입 시 기본 경로). 테스트는 항상 llm_call을 주입해 우회한다."""
    import config
    from openai import OpenAI

    client = OpenAI(api_key=getattr(config, "OPENAI_API_KEY", ""))
    model = getattr(config, "NEWS_LLM_CHEAP", "gpt-4-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def _grounding_candidates(draft: dict) -> List[dict]:
    """근거 판정 대상 문장: 4개 섹션 소속이면서 market_data 태그가 아닌 문장만."""
    return [
        s for s in draft.get("sentences", [])
        if s.get("section") in SECTIONS and s.get("article_ids") != [MARKET_DATA_TAG]
    ]


def _format_articles_for_grounding(articles: List[dict]) -> str:
    if not articles:
        return "(제공된 기사 없음)"
    lines = []
    for a in articles:
        lines.append(f"[기사 {a.get('id')}] {a.get('title', '')}\n본문: {a.get('body', '')}")
    return "\n\n".join(lines)


def _format_grounding_prompt(candidates: List[dict], articles: List[dict]) -> str:
    sentences_block = "\n".join(
        f"[idx {s.get('idx')}] ({s.get('section')}) {s.get('text')} — 인용 기사: {s.get('article_ids')}"
        for s in candidates
    )
    return f"""=== 판정 대상 문장 ===
{sentences_block}

=== 기사 풀 ===
{_format_articles_for_grounding(articles)}

각 문장이 인용된 기사에 의해 뒷받침되는지 판정해 JSON 배열로 응답하세요."""


def _parse_grounding_response(raw: str) -> List[dict]:
    """grounding 응답(JSON 배열) 파싱. 스키마 위반 시 ValueError."""
    raw = (raw or "").strip()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("응답에서 JSON 배열을 찾을 수 없습니다")
        data = json.loads(raw[start:end + 1])

    if not isinstance(data, list):
        raise ValueError("grounding 응답은 JSON 배열이어야 합니다")
    for item in data:
        if not isinstance(item, dict) or "idx" not in item or "grounded" not in item:
            raise ValueError("grounding 응답 항목 스키마 위반")
    return data


def check_grounding(
    draft: dict,
    articles: List[dict],
    llm_call: Optional[LlmCall] = None,
) -> List[dict]:
    """저가 LLM 배치 1콜로 문장별 근거(entailment)를 판정한다.
    파싱 실패 시 1회 재요청, 재실패하면 판정 대상 문장을 모두 grounded=False로 안전 처리한다
    (예외를 던지지 않음 — 게이트가 soft-fail로 흡수)."""
    call = llm_call or _default_cheap_llm_call

    candidates = _grounding_candidates(draft)
    if not candidates:
        return []

    user_prompt = _format_grounding_prompt(candidates, articles)

    results: Optional[List[dict]] = None
    prompt = user_prompt
    for attempt in range(2):
        raw = call(_GROUNDING_SYSTEM_PROMPT, prompt)
        try:
            results = _parse_grounding_response(raw)
            break
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("grounding 응답 파싱 실패 (attempt=%d): %s", attempt + 1, e)
            prompt = prompt + _GROUNDING_RETRY_NOTICE

    if results is None:
        results = [
            {"idx": s.get("idx"), "grounded": False, "reason": "판정 응답 파싱 실패로 안전 처리"}
            for s in candidates
        ]

    return [
        {"idx": r.get("idx"), "type": "grounding", "reason": r.get("reason", "")}
        for r in results
        if r.get("grounded") is False
    ]


def _get_whitelist():
    """도메인 화이트리스트 조회 (sources.py US-003 완료 전 지연 임포트).
    테스트는 이 함수를 monkeypatch해서 sources.py 존재 여부와 무관하게 독립 실행한다."""
    from news_briefing.sources import FOREIGN_ALLOWED_DOMAINS, KR_ALLOWED_DOMAINS
    return set(FOREIGN_ALLOWED_DOMAINS) | set(KR_ALLOWED_DOMAINS)


def _domain_allowed(domain: Optional[str], whitelist) -> bool:
    """domain이 화이트리스트 도메인 자신이거나 그 서브도메인(예: www.cnbc.com → cnbc.com)이면 허용.
    실제 기사 URL의 netloc은 www. 등 서브도메인을 포함하는 경우가 많아 정확 일치만으로는
    화이트리스트에 있는 도메인의 기사가 전부 오탐(hard violation)으로 걸리는 문제가 있었다."""
    if not domain:
        return False
    return any(domain == wd or domain.endswith("." + wd) for wd in whitelist)


def check_links(articles: List[dict]) -> List[dict]:
    """기사 도메인이 화이트리스트(서브도메인 포함)에 속하는지, HTTP 상태가 200~399 범위인지 확인한다.
    http_status가 None이면 '미검증'으로 보고 hard violation으로 취급하지 않는다(건너뜀)."""
    whitelist = _get_whitelist()
    violations = []
    for a in articles or []:
        domain = a.get("domain")
        status = a.get("http_status")
        article_id = a.get("id")

        if not _domain_allowed(domain, whitelist):
            violations.append({
                "idx": None,
                "type": "link",
                "reason": f"화이트리스트에 없는 도메인: {domain}",
                "article_id": article_id,
            })
            continue

        if status is None:
            continue  # 미검증 — hard violation 아님

        if not (200 <= status < 400):
            violations.append({
                "idx": None,
                "type": "link",
                "reason": f"HTTP 상태 코드 이상: {status}",
                "article_id": article_id,
            })

    return violations
