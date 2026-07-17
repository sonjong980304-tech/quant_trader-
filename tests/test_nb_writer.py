"""
tests/test_nb_writer.py — news_briefing.writer TDD (closed-book 브리핑 작성)

플랜: .omc/plans/news-briefing-revamp-plan.md §5 6단계.
write_briefing/find_uncited/rewrite/parse_draft를 검증한다.
실제 OpenAI API 호출 없음 — 모든 테스트는 llm_call을 mock 함수로 주입한다.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from news_briefing.writer import write_briefing, find_uncited, rewrite, parse_draft


# ─────────────────────────────────────────────
# 공통 픽스처
# ─────────────────────────────────────────────

ARTICLES = [
    {"id": 1, "url": "https://reuters.com/a", "domain": "reuters.com",
     "title": "나스닥 급등", "body": "나스닥이 기술주 강세로 1.5% 상승했다."},
    {"id": 2, "url": "https://cnbc.com/b", "domain": "cnbc.com",
     "title": "연준 발언", "body": "연준 위원이 금리 동결을 시사했다."},
]

MARKET_SNAPSHOT = {
    "nasdaq": {"close": 18000.0, "change_pct": 1.5},
    "sp500": {"close": 5800.0, "change_pct": 0.8},
}

STOCK_NEWS = [
    {"ticker": "005930", "article_id": 1, "category": "event", "summary": "삼성전자 관련 뉴스"},
]


def _valid_draft_json():
    return json.dumps({
        "sentences": [
            {"idx": 0, "section": "nasdaq", "text": "나스닥은 1.5% 상승 마감했다.",
             "article_ids": ["market_data"]},
            {"idx": 1, "section": "cause", "text": "기술주 강세와 연준의 금리 동결 시사가 상승을 이끌었다.",
             "article_ids": [1, 2]},
            {"idx": 2, "section": "kr_impact", "text": "미국 증시 강세는 국내 증시에도 우호적일 전망이다.",
             "article_ids": [1]},
            {"idx": 3, "section": "holdings", "text": "삼성전자 관련 뉴스가 있었다.",
             "article_ids": [1]},
        ],
        "forecasts": [
            {"market": "KOSPI", "direction": "up", "rationale": "미국 증시 강세 동반 상승 기대"},
            {"market": "KOSDAQ", "direction": "flat", "rationale": "기술주 혼조 영향으로 보합 전망"},
        ],
    }, ensure_ascii=False)


# ─────────────────────────────────────────────
# write_briefing — 정상 draft 파싱
# ─────────────────────────────────────────────

class TestWriteBriefing:
    def test_parses_valid_llm_response(self):
        calls = []

        def fake_llm_call(system, user):
            calls.append((system, user))
            return _valid_draft_json()

        draft = write_briefing(ARTICLES, MARKET_SNAPSHOT, STOCK_NEWS, llm_call=fake_llm_call)

        assert len(calls) == 1   # 정상경로 1콜
        assert len(draft["sentences"]) == 4
        assert len(draft["forecasts"]) == 2
        assert draft["sentences"][0]["section"] == "nasdaq"
        assert {f["market"] for f in draft["forecasts"]} == {"KOSPI", "KOSDAQ"}

    def test_system_prompt_states_closed_book_rules(self):
        captured = {}

        def fake_llm_call(system, user):
            captured["system"] = system
            captured["user"] = user
            return _valid_draft_json()

        write_briefing(ARTICLES, MARKET_SNAPSHOT, STOCK_NEWS, llm_call=fake_llm_call)

        assert "제공된 기사와 시장 데이터만 근거로 작성" in captured["system"]
        assert "article_ids" in captured["system"]
        assert "기사에 없는 내용" in captured["system"]
        assert "그대로 사용" in captured["system"]

    def test_parse_failure_retries_once_then_raises(self):
        calls = []

        def bad_llm_call(system, user):
            calls.append(user)
            return "이건 JSON이 아닙니다"

        with pytest.raises(ValueError):
            write_briefing(ARTICLES, MARKET_SNAPSHOT, STOCK_NEWS, llm_call=bad_llm_call)

        assert len(calls) == 2   # 최초 1회 + 재요청 1회


# ─────────────────────────────────────────────
# parse_draft — 스키마 위반 검출
# ─────────────────────────────────────────────

class TestParseDraft:
    def test_valid_json_parses(self):
        draft = parse_draft(_valid_draft_json())
        assert len(draft["sentences"]) == 4

    def test_rejects_unknown_section(self):
        raw = json.dumps({
            "sentences": [{"idx": 0, "section": "weather", "text": "x", "article_ids": [1]}],
            "forecasts": [
                {"market": "KOSPI", "direction": "up", "rationale": "r"},
                {"market": "KOSDAQ", "direction": "flat", "rationale": "r"},
            ],
        })
        with pytest.raises(ValueError):
            parse_draft(raw)

    def test_rejects_non_sequential_idx(self):
        raw = json.dumps({
            "sentences": [
                {"idx": 0, "section": "nasdaq", "text": "x", "article_ids": ["market_data"]},
                {"idx": 2, "section": "cause", "text": "y", "article_ids": [1]},
            ],
            "forecasts": [
                {"market": "KOSPI", "direction": "up", "rationale": "r"},
                {"market": "KOSDAQ", "direction": "flat", "rationale": "r"},
            ],
        })
        with pytest.raises(ValueError):
            parse_draft(raw)

    def test_rejects_unknown_direction(self):
        raw = json.dumps({
            "sentences": [{"idx": 0, "section": "nasdaq", "text": "x", "article_ids": ["market_data"]}],
            "forecasts": [
                {"market": "KOSPI", "direction": "sideways", "rationale": "r"},
                {"market": "KOSDAQ", "direction": "flat", "rationale": "r"},
            ],
        })
        with pytest.raises(ValueError):
            parse_draft(raw)

    def test_rejects_missing_market(self):
        raw = json.dumps({
            "sentences": [{"idx": 0, "section": "nasdaq", "text": "x", "article_ids": ["market_data"]}],
            "forecasts": [
                {"market": "KOSPI", "direction": "up", "rationale": "r"},
            ],
        })
        with pytest.raises(ValueError):
            parse_draft(raw)

    def test_extracts_json_from_code_fence(self):
        raw = "```json\n" + _valid_draft_json() + "\n```"
        draft = parse_draft(raw)
        assert len(draft["sentences"]) == 4


# ─────────────────────────────────────────────
# find_uncited — 인용 누락 검출 + market_data 규약
# ─────────────────────────────────────────────

class TestFindUncited:
    def test_detects_sentence_without_article_ids(self):
        draft = {
            "sentences": [
                {"idx": 0, "section": "nasdaq", "text": "x", "article_ids": []},
                {"idx": 1, "section": "cause", "text": "y", "article_ids": [1]},
            ],
            "forecasts": [],
        }
        assert find_uncited(draft) == [0]

    def test_market_data_tag_counts_as_cited(self):
        draft = {
            "sentences": [
                {"idx": 0, "section": "nasdaq", "text": "지수 서술", "article_ids": ["market_data"]},
            ],
            "forecasts": [],
        }
        assert find_uncited(draft) == []

    def test_ignores_sentences_outside_citable_sections(self):
        draft = {
            "sentences": [
                {"idx": 0, "section": "intro", "text": "x", "article_ids": []},
            ],
            "forecasts": [],
        }
        assert find_uncited(draft) == []


# ─────────────────────────────────────────────
# rewrite — feedback 반영
# ─────────────────────────────────────────────

class TestRewrite:
    def test_includes_feedback_in_prompt(self):
        captured = {}

        def fake_llm_call(system, user):
            captured["user"] = user
            return _valid_draft_json()

        draft = parse_draft(_valid_draft_json())
        feedback = [{"idx": 1, "reason": "수치가 시장 데이터와 불일치"}]

        rewrite(draft, ARTICLES, MARKET_SNAPSHOT, feedback, llm_call=fake_llm_call)

        assert "idx 1" in captured["user"]
        assert "수치가 시장 데이터와 불일치" in captured["user"]

    def test_returns_valid_parsed_draft(self):
        def fake_llm_call(system, user):
            return _valid_draft_json()

        draft = parse_draft(_valid_draft_json())
        result = rewrite(draft, ARTICLES, MARKET_SNAPSHOT,
                          [{"idx": 0, "reason": "r"}], llm_call=fake_llm_call)

        assert len(result["sentences"]) == 4
