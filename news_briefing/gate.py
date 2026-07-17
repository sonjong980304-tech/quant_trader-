"""
news_briefing/gate.py — 브리핑 초안 검증 상태머신
(플랜: .omc/plans/news-briefing-revamp-plan.md §3·§5 8단계)

writer.py가 만든 draft를 verifier.py로 검증하고, hard violation(수치 모순·화이트리스트
링크 위반)이 있으면 writer.rewrite로 재생성을 요청한다(최대 max_regen회). 그래도 남으면
해당 문장을 draft에서 제거한다. soft violation(grounding 미충족)은 제거하지 않고
warn_flags로 누적한다(게이트가 흡수).

사실 검증: verifier.extract_market_claims로 market_data 태그 문장에서 지수 등락률
클레임을 정규식 추출해 check_facts로 실측치와 대조한다(방향이 모호한 문장은 안전하게
건너뜀 — verifier.extract_market_claims 참조).
"""
import logging
from typing import Callable, List, Optional

from news_briefing import verifier, writer
from news_briefing.writer import MARKET_DATA_TAG, SECTIONS

logger = logging.getLogger(__name__)

LlmCall = Callable[[str, str], str]


def _map_link_violations(link_violations: List[dict], sentences: List[dict]) -> List[dict]:
    """check_links의 article_id 기반 위반을, 그 기사를 인용한 문장 idx로 역매핑한다.
    같은 article_id를 여러 문장이 인용하면 전부 hard violation 대상이 된다.
    인용한 문장이 없으면 idx=None을 유지한다(재생성 피드백·제거 대상에서 자연히 제외)."""
    mapped = []
    for v in link_violations:
        article_id = v.get("article_id")
        matched_idxs = [
            s.get("idx") for s in sentences
            if article_id in (s.get("article_ids") or [])
        ]
        if not matched_idxs:
            mapped.append({"idx": None, "type": "link", "reason": v.get("reason", "")})
            continue
        for idx in matched_idxs:
            mapped.append({"idx": idx, "type": "link", "reason": v.get("reason", "")})
    return mapped


def _verify(draft: dict, articles: List[dict], market_snapshot: dict, llm_call: Optional[LlmCall]):
    """한 차례 검증 패스: grounding(soft) + links(hard, idx 역매핑) + facts(hard, 현재는 no-op).
    반환: (hard_violations, soft_violations)"""
    soft_violations = verifier.check_grounding(draft, articles, llm_call)

    link_violations = verifier.check_links(articles)
    hard_violations = _map_link_violations(link_violations, draft.get("sentences", []))

    claims = verifier.extract_market_claims(draft)
    fact_violations = verifier.check_facts(claims, market_snapshot)
    hard_violations = hard_violations + fact_violations

    return hard_violations, soft_violations


def run_gate(
    draft: dict,
    articles: List[dict],
    market_snapshot: dict,
    llm_call: Optional[LlmCall] = None,
    max_regen: int = 2,
) -> dict:
    """draft를 검증→(hard violation 있으면) 재생성→(그래도 남으면) 제거하는 게이트 상태머신."""
    regen_count = 0
    hard_violations, soft_violations = _verify(draft, articles, market_snapshot, llm_call)

    while hard_violations and regen_count < max_regen:
        feedback = [{"idx": v["idx"], "reason": v["reason"]} for v in hard_violations]
        draft = writer.rewrite(draft, articles, market_snapshot, feedback, llm_call)
        regen_count += 1
        hard_violations, soft_violations = _verify(draft, articles, market_snapshot, llm_call)

    verified_sentences = draft.get("sentences", [])
    total_sentences = len(verified_sentences)
    hard_count = len(hard_violations)

    if hard_violations:
        remove_idxs = {v["idx"] for v in hard_violations if v.get("idx") is not None}
        draft["sentences"] = [s for s in verified_sentences if s.get("idx") not in remove_idxs]

    warn_flags = sorted({v["idx"] for v in soft_violations if v.get("idx") is not None})

    fact_score = max(0.0, 1.0 - (hard_count / total_sentences)) if total_sentences else 1.0

    grounding_targets = [
        s for s in verified_sentences
        if s.get("section") in SECTIONS and s.get("article_ids") != [MARKET_DATA_TAG]
    ]
    grounding_target_count = len(grounding_targets)
    grounding_score = (
        max(0.0, 1.0 - (len(soft_violations) / grounding_target_count))
        if grounding_target_count else 1.0
    )

    return {
        "final_draft": draft,
        "fact_score": fact_score,
        "grounding_score": grounding_score,
        "regen_count": regen_count,
        "warn_flags": warn_flags,
    }
