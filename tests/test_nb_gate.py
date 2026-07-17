"""
tests/test_nb_gate.py — news_briefing.gate TDD (검증→재생성→제거 상태머신)

플랜: .omc/plans/news-briefing-revamp-plan.md §3·§5 8단계.
run_gate가 verifier.check_grounding/check_links/check_facts와 writer.rewrite를 어떻게
조합하는지 검증한다. verifier.*/writer.rewrite는 전부 monkeypatch로 대체하며 실제
OpenAI API 호출은 없다.

gate는 verifier.extract_market_claims로 market_data 태그 문장에서 지수 등락률 클레임을
추출해 check_facts에 실제로 전달한다(test_extracts_and_passes_real_claims_to_check_facts).
extract_market_claims 자체의 상세 케이스는 tests/test_nb_verifier.py에서 검증한다.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import news_briefing.verifier as verifier
import news_briefing.writer as writer
from news_briefing.gate import run_gate


# ─────────────────────────────────────────────
# 공통 픽스처
# ─────────────────────────────────────────────

ARTICLES = [
    {"id": 1, "url": "https://reuters.com/a", "domain": "reuters.com", "http_status": 200},
    {"id": 2, "url": "https://evil.com/b", "domain": "evil.com", "http_status": 200},
]

MARKET_SNAPSHOT = {
    "nasdaq": {"close": 18000.0, "change_pct": 1.5},
}


def _make_draft():
    return {
        "sentences": [
            {"idx": 0, "section": "nasdaq", "text": "나스닥은 1.5% 상승 마감했다.",
             "article_ids": ["market_data"]},
            {"idx": 1, "section": "cause", "text": "기술주 강세가 상승을 이끌었다.",
             "article_ids": [1]},
            {"idx": 2, "section": "kr_impact", "text": "국내 증시도 영향을 받을 전망이다.",
             "article_ids": [2]},
        ],
        "forecasts": [
            {"market": "KOSPI", "direction": "up", "rationale": "r"},
            {"market": "KOSDAQ", "direction": "flat", "rationale": "r"},
        ],
    }


# ─────────────────────────────────────────────
# 1. 통과 — 위반 없음
# ─────────────────────────────────────────────

class TestGatePass:
    def test_no_violations_passes_through_unchanged(self, monkeypatch):
        draft = _make_draft()
        rewrite_calls = []

        monkeypatch.setattr(verifier, "check_grounding", lambda d, a, llm_call=None: [])
        monkeypatch.setattr(verifier, "check_links", lambda a: [])
        monkeypatch.setattr(verifier, "check_facts", lambda claims, snap: [])
        monkeypatch.setattr(writer, "rewrite", lambda *args, **kwargs: rewrite_calls.append(1))

        result = run_gate(draft, ARTICLES, MARKET_SNAPSHOT)

        assert result["regen_count"] == 0
        assert result["final_draft"] == draft
        assert result["fact_score"] == 1.0
        assert result["grounding_score"] == 1.0
        assert result["warn_flags"] == []
        assert rewrite_calls == []  # 재생성 호출 없음

    def test_extracts_and_passes_real_claims_to_check_facts(self, monkeypatch):
        """market_data 태그 문장("나스닥은 1.5% 상승 마감했다")에서 실제로 클레임을
        추출해 check_facts에 전달한다(예전의 claims=[] no-op에서 벗어남)."""
        draft = _make_draft()
        captured = {}

        monkeypatch.setattr(verifier, "check_grounding", lambda d, a, llm_call=None: [])
        monkeypatch.setattr(verifier, "check_links", lambda a: [])

        def fake_check_facts(claims, snap):
            captured["claims"] = claims
            captured["snap"] = snap
            return []

        monkeypatch.setattr(verifier, "check_facts", fake_check_facts)

        run_gate(draft, ARTICLES, MARKET_SNAPSHOT)

        assert captured["claims"] == [{"idx": 0, "market": "nasdaq", "claimed_pct": 1.5}]
        assert captured["snap"] == MARKET_SNAPSHOT

    def test_fact_violation_from_real_claim_triggers_regen(self, monkeypatch):
        """market_snapshot과 어긋나는 수치 클레임("나스닥은 9.9% 상승")은 실제
        check_facts(진짜 verifier 로직)를 통해 hard violation으로 잡혀 재생성을 유발한다."""
        draft = {
            "sentences": [
                {"idx": 0, "section": "nasdaq", "text": "나스닥은 9.9% 상승 마감했다.",
                 "article_ids": ["market_data"]},
            ],
            "forecasts": [
                {"market": "KOSPI", "direction": "up", "rationale": "r"},
                {"market": "KOSDAQ", "direction": "flat", "rationale": "r"},
            ],
        }
        fixed_draft = {
            "sentences": [
                {"idx": 0, "section": "nasdaq", "text": "나스닥은 1.5% 상승 마감했다.",
                 "article_ids": ["market_data"]},
            ],
            "forecasts": draft["forecasts"],
        }

        monkeypatch.setattr(verifier, "check_grounding", lambda d, a, llm_call=None: [])
        monkeypatch.setattr(verifier, "check_links", lambda a: [])
        # check_facts는 monkeypatch하지 않음 — 실제 verifier.check_facts + extract_market_claims 사용
        monkeypatch.setattr(writer, "rewrite", lambda *a, **k: fixed_draft)

        result = run_gate(draft, ARTICLES, MARKET_SNAPSHOT)

        assert result["regen_count"] == 1
        assert result["final_draft"] == fixed_draft
        assert result["fact_score"] == 1.0


# ─────────────────────────────────────────────
# 2. 1차실패→재생성→통과
# ─────────────────────────────────────────────

class TestGateRegenerateThenPass:
    def test_one_regen_then_pass(self, monkeypatch):
        draft = _make_draft()
        rewritten = _make_draft()
        rewritten["_marker"] = "rewritten"

        link_calls = []

        def fake_check_links(articles):
            link_calls.append(1)
            if len(link_calls) == 1:
                return [{"idx": None, "type": "link", "reason": "화이트리스트에 없는 도메인",
                          "article_id": 2}]
            return []

        rewrite_calls = []

        def fake_rewrite(d, a, snap, feedback, llm_call=None):
            rewrite_calls.append(feedback)
            return rewritten

        monkeypatch.setattr(verifier, "check_grounding", lambda d, a, llm_call=None: [])
        monkeypatch.setattr(verifier, "check_links", fake_check_links)
        monkeypatch.setattr(verifier, "check_facts", lambda claims, snap: [])
        monkeypatch.setattr(writer, "rewrite", fake_rewrite)

        result = run_gate(draft, ARTICLES, MARKET_SNAPSHOT)

        assert result["regen_count"] == 1
        assert result["final_draft"] is rewritten
        assert len(rewrite_calls) == 1
        # article_id=2를 인용한 문장(idx=2)으로 역매핑된 피드백이 rewrite에 전달됨
        assert rewrite_calls[0] == [{"idx": 2, "reason": "화이트리스트에 없는 도메인"}]


# ─────────────────────────────────────────────
# 3. 2회실패→제거
# ─────────────────────────────────────────────

class TestGateExhaustsRegenThenRemoves:
    def test_max_regen_exhausted_removes_violating_sentence(self, monkeypatch):
        draft = _make_draft()

        def fake_check_links(articles):
            # 재생성해도 동일한 위반이 계속 발생 (article_id=2, evil.com)
            return [{"idx": None, "type": "link", "reason": "화이트리스트에 없는 도메인",
                      "article_id": 2}]

        rewrite_calls = []

        def fake_rewrite(d, a, snap, feedback, llm_call=None):
            rewrite_calls.append(feedback)
            return _make_draft()  # 매번 동일한 3문장 draft를 새로 반환

        monkeypatch.setattr(verifier, "check_grounding", lambda d, a, llm_call=None: [])
        monkeypatch.setattr(verifier, "check_links", fake_check_links)
        monkeypatch.setattr(verifier, "check_facts", lambda claims, snap: [])
        monkeypatch.setattr(writer, "rewrite", fake_rewrite)

        result = run_gate(draft, ARTICLES, MARKET_SNAPSHOT, max_regen=2)

        assert result["regen_count"] == 2
        assert len(rewrite_calls) == 2
        final_idxs = {s["idx"] for s in result["final_draft"]["sentences"]}
        assert 2 not in final_idxs  # 화이트리스트 위반 문장(idx=2) 제거됨
        assert final_idxs == {0, 1}
        assert result["fact_score"] < 1.0


# ─────────────────────────────────────────────
# 4. soft플래그 — grounding 미충족은 제거하지 않고 warn_flags로 누적
# ─────────────────────────────────────────────

class TestGateSoftFlag:
    def test_grounding_violation_flagged_not_removed(self, monkeypatch):
        draft = _make_draft()
        rewrite_calls = []

        monkeypatch.setattr(
            verifier, "check_grounding",
            lambda d, a, llm_call=None: [{"idx": 1, "type": "grounding", "reason": "근거 부족"}],
        )
        monkeypatch.setattr(verifier, "check_links", lambda a: [])
        monkeypatch.setattr(verifier, "check_facts", lambda claims, snap: [])
        monkeypatch.setattr(writer, "rewrite", lambda *args, **kwargs: rewrite_calls.append(1))

        result = run_gate(draft, ARTICLES, MARKET_SNAPSHOT)

        assert result["regen_count"] == 0
        assert rewrite_calls == []  # soft violation은 재생성을 유발하지 않음
        assert result["warn_flags"] == [1]
        final_idxs = {s["idx"] for s in result["final_draft"]["sentences"]}
        assert 1 in final_idxs  # 제거되지 않음
        assert result["grounding_score"] < 1.0
        assert result["fact_score"] == 1.0
