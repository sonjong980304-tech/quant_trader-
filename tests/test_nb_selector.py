"""
tests/test_nb_selector.py
news_briefing/selector.py 검증 — 저가 LLM 1콜로 매크로/종목 뉴스 선별 + 카테고리 라벨링

실행: pytest tests/test_nb_selector.py -v
"""

import json
import os
import sys

import pytest

# ─── 경로 설정 ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from news_briefing.selector import select_articles


# ─── 공통 픽스처 (데이터 계약 준수) ─────────────────────────────────────────

def _articles():
    return [
        {"id": 1, "url": "https://reuters.com/a1", "domain": "reuters.com",
         "source_lang": "en", "title": "Nasdaq falls on rate fears",
         "body": "The Nasdaq Composite fell 1.2% amid rate hike concerns...",
         "published_at": "2026-07-10T05:00:00Z"},
        {"id": 2, "url": "https://hankyung.com/a2", "domain": "hankyung.com",
         "source_lang": "ko", "title": "삼성전자 신제품 발표",
         "body": "삼성전자가 신형 반도체 라인업을 발표했다...",
         "published_at": "2026-07-10T06:00:00Z"},
        {"id": 3, "url": "https://cnbc.com/a3", "domain": "cnbc.com",
         "source_lang": "en", "title": "Fed signals pause",
         "body": "Federal Reserve officials signaled a pause in rate hikes...",
         "published_at": "2026-07-10T07:00:00Z"},
    ]


def _holdings():
    return [
        {"ticker": "005930", "name": "삼성전자", "source": "kis"},
        {"ticker": "AAPL", "name": "Apple", "source": "paper"},
    ]


def _make_llm_call(responses):
    """호출될 때마다 responses 리스트에서 순서대로 값을 반환하는 mock. 호출 횟수를 기록."""
    calls = {"count": 0}

    def _call(system, user):
        calls["count"] += 1
        idx = min(calls["count"] - 1, len(responses) - 1)
        return responses[idx]

    return _call, calls


# ─────────────────────────────────────────────────────────────────────────────
# 1. 정상 JSON 선별
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalSelection:
    def test_parses_valid_json_response(self):
        valid = json.dumps({
            "macro_article_ids": [1, 3],
            "stock_news": [
                {"ticker": "005930", "article_id": 2, "category": "event",
                 "summary": "신제품 발표로 주가 영향 예상"},
            ],
        }, ensure_ascii=False)
        llm_call, calls = _make_llm_call([valid])

        result = select_articles(_articles(), _holdings(), llm_call=llm_call)

        assert result["macro_article_ids"] == [1, 3]
        assert len(result["stock_news"]) == 1
        assert result["stock_news"][0]["ticker"] == "005930"
        assert result["stock_news"][0]["article_id"] == 2
        assert calls["count"] == 1

    def test_llm_call_receives_system_and_user_prompts(self):
        """llm_call(system, user) 시그니처 — system/user 프롬프트가 모두 전달되는지"""
        valid = json.dumps({"macro_article_ids": [], "stock_news": []})
        captured = {}

        def _call(system, user):
            captured["system"] = system
            captured["user"] = user
            return valid

        select_articles(_articles(), _holdings(), llm_call=_call)

        assert captured["system"]
        assert captured["user"]
        # 본문 전체가 아니라 id·제목만 전달돼야 함 (토큰 절약)
        assert "Nasdaq falls on rate fears" in captured["user"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. 카테고리 라벨 파싱 (event / cause / risk)
# ─────────────────────────────────────────────────────────────────────────────

class TestCategoryLabelParsing:
    def test_all_three_categories_parsed(self):
        valid = json.dumps({
            "macro_article_ids": [],
            "stock_news": [
                {"ticker": "005930", "article_id": 1, "category": "event", "summary": "s1"},
                {"ticker": "AAPL", "article_id": 2, "category": "cause", "summary": "s2"},
                {"ticker": "005930", "article_id": 3, "category": "risk", "summary": "s3"},
            ],
        }, ensure_ascii=False)
        llm_call, _ = _make_llm_call([valid])

        result = select_articles(_articles(), _holdings(), llm_call=llm_call)

        categories = [item["category"] for item in result["stock_news"]]
        assert categories == ["event", "cause", "risk"]

    def test_invalid_category_is_dropped(self):
        valid = json.dumps({
            "macro_article_ids": [],
            "stock_news": [
                {"ticker": "005930", "article_id": 1, "category": "event", "summary": "s1"},
                {"ticker": "AAPL", "article_id": 2, "category": "not_a_category", "summary": "s2"},
            ],
        }, ensure_ascii=False)
        llm_call, _ = _make_llm_call([valid])

        result = select_articles(_articles(), _holdings(), llm_call=llm_call)

        assert len(result["stock_news"]) == 1
        assert result["stock_news"][0]["category"] == "event"


# ─────────────────────────────────────────────────────────────────────────────
# 3. 총합 12건 초과 시 절단
# ─────────────────────────────────────────────────────────────────────────────

class TestTruncation:
    def test_truncates_to_12_preserving_order(self):
        macro_ids = list(range(1, 6))          # 5건
        stock_news = [
            {"ticker": "005930", "article_id": 100 + i, "category": "event", "summary": f"s{i}"}
            for i in range(10)                   # 10건 → 총 15건
        ]
        valid = json.dumps({"macro_article_ids": macro_ids, "stock_news": stock_news})
        llm_call, _ = _make_llm_call([valid])

        result = select_articles(_articles(), _holdings(), llm_call=llm_call)

        total = len(result["macro_article_ids"]) + len(result["stock_news"])
        assert total == 12
        # LLM이 준 순서대로 절단: macro 5건 전부 유지 + stock 앞에서부터 7건만
        assert result["macro_article_ids"] == macro_ids
        assert len(result["stock_news"]) == 7
        assert [s["article_id"] for s in result["stock_news"]] == [100 + i for i in range(7)]

    def test_no_truncation_when_at_or_under_limit(self):
        macro_ids = list(range(1, 7))           # 6건
        stock_news = [
            {"ticker": "005930", "article_id": 200 + i, "category": "risk", "summary": f"s{i}"}
            for i in range(6)                    # 6건 → 총 12건 (한도 정확히)
        ]
        valid = json.dumps({"macro_article_ids": macro_ids, "stock_news": stock_news})
        llm_call, _ = _make_llm_call([valid])

        result = select_articles(_articles(), _holdings(), llm_call=llm_call)

        assert len(result["macro_article_ids"]) == 6
        assert len(result["stock_news"]) == 6


# ─────────────────────────────────────────────────────────────────────────────
# 4. 파싱 실패 시 1회 재요청
# ─────────────────────────────────────────────────────────────────────────────

class TestRetryOnParseFailure:
    def test_retries_once_and_succeeds(self):
        invalid = "이건 JSON이 아닙니다"
        valid = json.dumps({"macro_article_ids": [1], "stock_news": []})
        llm_call, calls = _make_llm_call([invalid, valid])

        result = select_articles(_articles(), _holdings(), llm_call=llm_call)

        assert calls["count"] == 2
        assert result["macro_article_ids"] == [1]

    def test_retries_once_on_malformed_schema(self):
        """JSON이지만 계약을 위반한 경우도 파싱 실패로 취급해 재요청."""
        malformed = json.dumps({"macro_article_ids": "not-a-list", "stock_news": []})
        valid = json.dumps({"macro_article_ids": [2], "stock_news": []})
        llm_call, calls = _make_llm_call([malformed, valid])

        result = select_articles(_articles(), _holdings(), llm_call=llm_call)

        assert calls["count"] == 2
        assert result["macro_article_ids"] == [2]


# ─────────────────────────────────────────────────────────────────────────────
# 5. 재요청도 실패하면 ValueError
# ─────────────────────────────────────────────────────────────────────────────

class TestRetryExhaustedRaises:
    def test_raises_value_error_after_two_failures(self):
        invalid1 = "not json at all"
        invalid2 = "{broken json"
        llm_call, calls = _make_llm_call([invalid1, invalid2])

        with pytest.raises(ValueError):
            select_articles(_articles(), _holdings(), llm_call=llm_call)

        assert calls["count"] == 2   # 최대 2회(최초+재요청 1회)까지만 시도
