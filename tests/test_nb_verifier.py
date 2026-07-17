"""
tests/test_nb_verifier.py — news_briefing.verifier TDD (사실 대조·근거 판정·링크 체크)

플랜: .omc/plans/news-briefing-revamp-plan.md §3·§5 7단계.
check_facts/check_grounding/check_links/_within_tolerance를 검증한다.
실제 OpenAI API 호출 없음 — check_grounding 테스트는 모두 llm_call을 mock 함수로 주입한다.
sources.py(US-003)는 아직 없을 수 있으므로 check_links는 _get_whitelist를 monkeypatch해서
독립적으로 검증한다.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import news_briefing.verifier as verifier
from news_briefing.verifier import (
    _within_tolerance,
    check_facts,
    check_grounding,
    check_links,
    extract_market_claims,
)


# ─────────────────────────────────────────────
# _within_tolerance — 허용오차 경계
# ─────────────────────────────────────────────

class TestWithinTolerance:
    def test_change_pct_within_tolerance(self):
        assert _within_tolerance(1.52, 1.50) is True  # diff 0.02 < 0.05

    def test_change_pct_exact_boundary(self):
        assert _within_tolerance(1.55, 1.50) is True  # diff 0.05 == 허용오차, 통과

    def test_change_pct_exceeds_tolerance(self):
        assert _within_tolerance(1.56, 1.50) is False  # diff 0.06 > 0.05

    def test_index_level_exact_boundary(self):
        assert _within_tolerance(18000.1, 18000.0, is_index_level=True) is True  # diff 0.1

    def test_index_level_exceeds_tolerance(self):
        assert _within_tolerance(18000.11, 18000.0, is_index_level=True) is False  # diff 0.11 > 0.1


# ─────────────────────────────────────────────
# check_facts — 클레임 vs market_snapshot 대조
# ─────────────────────────────────────────────

MARKET_SNAPSHOT = {
    "nasdaq": {"close": 18000.0, "change_pct": 1.50},
    "sp500": {"close": 5800.0, "change_pct": 0.80},
    "kospi": {"close": 2700.0, "change_pct": -0.30},
    "kosdaq": None,  # 조회 실패 케이스
}


class TestCheckFacts:
    def test_within_tolerance_no_violation(self):
        claims = [{"idx": 0, "market": "nasdaq", "claimed_pct": 1.52}]
        assert check_facts(claims, MARKET_SNAPSHOT) == []

    def test_exact_boundary_no_violation(self):
        claims = [{"idx": 0, "market": "sp500", "claimed_pct": 0.85}]  # diff 0.05
        assert check_facts(claims, MARKET_SNAPSHOT) == []

    def test_exceeds_tolerance_violation(self):
        claims = [{"idx": 3, "market": "kospi", "claimed_pct": 0.0}]  # diff 0.30
        result = check_facts(claims, MARKET_SNAPSHOT)
        assert len(result) == 1
        assert result[0]["idx"] == 3
        assert result[0]["type"] == "fact"
        assert isinstance(result[0]["reason"], str) and result[0]["reason"]

    def test_missing_market_data_skipped(self):
        claims = [{"idx": 0, "market": "kosdaq", "claimed_pct": 5.0}]  # snapshot None
        assert check_facts(claims, MARKET_SNAPSHOT) == []

    def test_unknown_market_key_skipped(self):
        claims = [{"idx": 0, "market": "dow", "claimed_pct": 1.0}]  # snapshot에 없는 시장
        assert check_facts(claims, MARKET_SNAPSHOT) == []

    def test_multiple_claims_mixed(self):
        claims = [
            {"idx": 0, "market": "nasdaq", "claimed_pct": 1.51},  # 통과
            {"idx": 1, "market": "kospi", "claimed_pct": 1.0},    # 위반 (diff 1.3)
        ]
        result = check_facts(claims, MARKET_SNAPSHOT)
        assert len(result) == 1
        assert result[0]["idx"] == 1


# ─────────────────────────────────────────────
# check_grounding — 배치 entailment 1콜 + 재요청/안전처리
# ─────────────────────────────────────────────

DRAFT = {
    "sentences": [
        {"idx": 0, "section": "nasdaq", "text": "나스닥은 1.5% 상승 마감했다.",
         "article_ids": ["market_data"]},
        {"idx": 1, "section": "cause", "text": "기술주 강세가 상승을 이끌었다.",
         "article_ids": [1]},
        {"idx": 2, "section": "kr_impact", "text": "국내 증시도 우호적 영향을 받을 전망이다.",
         "article_ids": [1]},
        {"idx": 3, "section": "holdings", "text": "삼성전자 관련 반등 뉴스가 있었다.",
         "article_ids": [2]},
    ],
    "forecasts": [
        {"market": "KOSPI", "direction": "up", "rationale": "r"},
        {"market": "KOSDAQ", "direction": "flat", "rationale": "r"},
    ],
}

GROUNDING_ARTICLES = [
    {"id": 1, "url": "https://reuters.com/a", "domain": "reuters.com",
     "title": "나스닥 급등", "body": "나스닥이 기술주 강세로 1.5% 상승했다."},
    {"id": 2, "url": "https://cnbc.com/b", "domain": "cnbc.com",
     "title": "삼성전자 반등", "body": "삼성전자 주가가 반등했다."},
]


class TestCheckGrounding:
    def test_all_grounded_no_violations(self):
        def fake_llm_call(system, user):
            return json.dumps([
                {"idx": 1, "grounded": True, "reason": "ok"},
                {"idx": 2, "grounded": True, "reason": "ok"},
                {"idx": 3, "grounded": True, "reason": "ok"},
            ])

        violations = check_grounding(DRAFT, GROUNDING_ARTICLES, llm_call=fake_llm_call)
        assert violations == []

    def test_partial_violations(self):
        def fake_llm_call(system, user):
            return json.dumps([
                {"idx": 1, "grounded": True, "reason": "ok"},
                {"idx": 2, "grounded": False, "reason": "기사에 없는 내용"},
                {"idx": 3, "grounded": True, "reason": "ok"},
            ])

        violations = check_grounding(DRAFT, GROUNDING_ARTICLES, llm_call=fake_llm_call)
        assert violations == [{"idx": 2, "type": "grounding", "reason": "기사에 없는 내용"}]

    def test_batches_into_single_call(self):
        calls = []

        def fake_llm_call(system, user):
            calls.append(user)
            return json.dumps([
                {"idx": 1, "grounded": True, "reason": "ok"},
                {"idx": 2, "grounded": True, "reason": "ok"},
                {"idx": 3, "grounded": True, "reason": "ok"},
            ])

        check_grounding(DRAFT, GROUNDING_ARTICLES, llm_call=fake_llm_call)
        assert len(calls) == 1  # 정상경로 1콜

    def test_excludes_market_data_tagged_sentence_from_prompt(self):
        captured = {}

        def fake_llm_call(system, user):
            captured["user"] = user
            return json.dumps([
                {"idx": 1, "grounded": True, "reason": "ok"},
                {"idx": 2, "grounded": True, "reason": "ok"},
                {"idx": 3, "grounded": True, "reason": "ok"},
            ])

        check_grounding(DRAFT, GROUNDING_ARTICLES, llm_call=fake_llm_call)
        assert "나스닥은 1.5% 상승 마감했다." not in captured["user"]  # market_data 태그 제외

    def test_parse_failure_retries_once_then_safe_fallback(self):
        calls = []

        def bad_llm_call(system, user):
            calls.append(user)
            return "이건 JSON이 아닙니다"

        violations = check_grounding(DRAFT, GROUNDING_ARTICLES, llm_call=bad_llm_call)

        assert len(calls) == 2  # 최초 1회 + 재요청 1회
        # 안전 처리: 판정 대상 문장(1,2,3) 전부 grounded=False로 간주 -> violation
        assert {v["idx"] for v in violations} == {1, 2, 3}
        assert all(v["type"] == "grounding" for v in violations)

    def test_no_candidates_skips_llm_call(self):
        only_market_data_draft = {
            "sentences": [
                {"idx": 0, "section": "nasdaq", "text": "x", "article_ids": ["market_data"]},
            ],
            "forecasts": [],
        }
        calls = []

        def fake_llm_call(system, user):
            calls.append(1)
            return "[]"

        violations = check_grounding(only_market_data_draft, GROUNDING_ARTICLES, llm_call=fake_llm_call)
        assert violations == []
        assert calls == []  # 판정 대상이 없으면 LLM 호출 자체를 생략


# ─────────────────────────────────────────────
# check_links — 화이트리스트·HTTP 상태 체크
# ─────────────────────────────────────────────

class TestCheckLinks:
    def test_whitelisted_domain_ok_status_no_violation(self, monkeypatch):
        monkeypatch.setattr(verifier, "_get_whitelist", lambda: {"reuters.com", "yna.co.kr"})
        articles = [{"id": 1, "domain": "reuters.com", "http_status": 200}]
        assert check_links(articles) == []

    def test_domain_not_whitelisted_violation(self, monkeypatch):
        monkeypatch.setattr(verifier, "_get_whitelist", lambda: {"reuters.com"})
        articles = [{"id": 1, "domain": "evil.com", "http_status": 200}]
        result = check_links(articles)
        assert len(result) == 1
        assert result[0]["idx"] is None
        assert result[0]["type"] == "link"
        assert result[0]["article_id"] == 1

    def test_subdomain_of_whitelisted_domain_is_allowed(self, monkeypatch):
        """실제 기사 URL의 netloc은 www. 등 서브도메인을 포함하는 경우가 많다
        (예: www.cnbc.com). 화이트리스트가 cnbc.com이면 서브도메인도 허용돼야 한다
        — 실제 E2E에서 전 기사가 오탐 hard violation으로 걸렸던 회귀 방지 테스트."""
        monkeypatch.setattr(verifier, "_get_whitelist", lambda: {"cnbc.com"})
        articles = [{"id": 1, "domain": "www.cnbc.com", "http_status": 200}]
        assert check_links(articles) == []

    def test_lookalike_domain_is_not_allowed(self, monkeypatch):
        """서브도메인 허용이 접미어 일치로 오탐하지 않는지 확인
        (예: notcnbc.com이 cnbc.com의 '서브도메인'으로 오인되면 안 됨)."""
        monkeypatch.setattr(verifier, "_get_whitelist", lambda: {"cnbc.com"})
        articles = [{"id": 1, "domain": "notcnbc.com", "http_status": 200}]
        result = check_links(articles)
        assert len(result) == 1
        assert result[0]["type"] == "link"

    def test_http_status_out_of_range_violation(self, monkeypatch):
        monkeypatch.setattr(verifier, "_get_whitelist", lambda: {"reuters.com"})
        articles = [{"id": 1, "domain": "reuters.com", "http_status": 404}]
        result = check_links(articles)
        assert len(result) == 1
        assert result[0]["type"] == "link"
        assert result[0]["article_id"] == 1

    def test_http_status_none_is_unverified_not_violation(self, monkeypatch):
        monkeypatch.setattr(verifier, "_get_whitelist", lambda: {"reuters.com"})
        articles = [{"id": 1, "domain": "reuters.com", "http_status": None}]
        assert check_links(articles) == []

    def test_boundary_status_200_and_399_pass_others_fail(self, monkeypatch):
        monkeypatch.setattr(verifier, "_get_whitelist", lambda: {"reuters.com"})
        articles = [
            {"id": 1, "domain": "reuters.com", "http_status": 200},
            {"id": 2, "domain": "reuters.com", "http_status": 399},
            {"id": 3, "domain": "reuters.com", "http_status": 400},
            {"id": 4, "domain": "reuters.com", "http_status": 199},
        ]
        result = check_links(articles)
        assert {v["article_id"] for v in result} == {3, 4}


# ─────────────────────────────────────────────
# extract_market_claims — market_data 문장 → 수치 클레임 추출
# ─────────────────────────────────────────────

def _md_draft(text, idx=0, section="nasdaq"):
    return {"sentences": [{"idx": idx, "section": section, "text": text, "article_ids": ["market_data"]}]}


class TestExtractMarketClaims:
    def test_single_market_up(self):
        draft = _md_draft("나스닥은 1.5% 상승 마감했다.")
        assert extract_market_claims(draft) == [{"idx": 0, "market": "nasdaq", "claimed_pct": 1.5}]

    def test_single_market_down(self):
        draft = _md_draft("나스닥은 0.8% 하락 마감했다.")
        assert extract_market_claims(draft) == [{"idx": 0, "market": "nasdaq", "claimed_pct": -0.8}]

    def test_explicit_sign_overrides_direction_words(self):
        draft = _md_draft("나스닥은 -0.8% 변동했다.")
        assert extract_market_claims(draft) == [{"idx": 0, "market": "nasdaq", "claimed_pct": -0.8}]

    def test_compound_sentence_multiple_markets_same_direction(self):
        draft = _md_draft("나스닥이 0.29%, S&P500이 0.42% 상승 마감했다.")
        claims = extract_market_claims(draft)
        by_market = {c["market"]: c["claimed_pct"] for c in claims}
        assert by_market["nasdaq"] == pytest.approx(0.29)
        assert by_market["sp500"] == pytest.approx(0.42)

    def test_ambiguous_mixed_direction_sentence_is_skipped(self):
        """한 문장에 상승·하락이 둘 다 등장하면 안전하게 추출을 건너뛴다."""
        draft = _md_draft("나스닥은 상승했고 다우는 0.5% 하락했다.")
        claims = extract_market_claims(draft)
        assert claims == []

    def test_non_market_data_sentence_ignored(self):
        """market_data 태그가 아닌(실제 기사 인용) 문장은 추출 대상이 아니다."""
        draft = {"sentences": [
            {"idx": 0, "section": "cause", "text": "나스닥은 1.5% 상승했다.", "article_ids": [1]},
        ]}
        assert extract_market_claims(draft) == []

    def test_no_number_no_claim(self):
        draft = _md_draft("나스닥이 상승 마감했다.")
        assert extract_market_claims(draft) == []

    def test_kospi_kosdaq_extracted(self):
        draft = _md_draft("코스피는 2.5% 상승, 코스닥은 5.5% 상승했다.", section="kr_impact")
        claims = extract_market_claims(draft)
        by_market = {c["market"]: c["claimed_pct"] for c in claims}
        assert by_market["kospi"] == pytest.approx(2.5)
        assert by_market["kosdaq"] == pytest.approx(5.5)
