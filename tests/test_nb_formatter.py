"""
tests/test_nb_formatter.py — news_briefing.formatter TDD (브리핑 HTML 포맷팅)

플랜: .omc/plans/news-briefing-revamp-plan.md §3·§5.
format_briefing_html/split_html_by_length/build_feedback_keyboard_payload를 검증한다.
"""
import os
import sys
from html.parser import HTMLParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from news_briefing.formatter import (
    build_feedback_keyboard_payload,
    format_briefing_html,
    split_html_by_length,
)
from news_briefing.constants import make_callback_data


# ─────────────────────────────────────────────
# 공통 픽스처
# ─────────────────────────────────────────────

def _draft(sentences=None, forecasts=None):
    return {
        "sentences": sentences if sentences is not None else [
            {"idx": 0, "section": "nasdaq", "text": "나스닥은 1.5% 상승 마감했다.",
             "article_ids": ["market_data"]},
            {"idx": 1, "section": "cause", "text": "기술주 강세가 상승을 이끌었다.",
             "article_ids": [1, 2]},
            {"idx": 2, "section": "kr_impact", "text": "국내 증시에도 우호적일 전망이다.",
             "article_ids": [1]},
            {"idx": 3, "section": "holdings", "text": "삼성전자 관련 뉴스가 있었다.",
             "article_ids": [1]},
        ],
        "forecasts": forecasts if forecasts is not None else [
            {"market": "KOSPI", "direction": "up", "rationale": "미국 증시 강세 동반 상승 기대"},
            {"market": "KOSDAQ", "direction": "flat", "rationale": "기술주 혼조 영향으로 보합 전망"},
        ],
    }


ARTICLES_BY_ID = {
    1: {"url": "https://reuters.com/a", "title": "나스닥 급등"},
    2: {"url": "https://cnbc.com/b", "title": "연준 발언"},
}


class _AnchorBalanceChecker(HTMLParser):
    """<a> 태그가 chunk 경계에서 절단되지 않았는지(열림/닫힘 균형) 검사한다."""

    def __init__(self):
        HTMLParser.__init__(self)
        self.open_anchors = 0

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.open_anchors += 1

    def handle_endtag(self, tag):
        if tag == "a":
            self.open_anchors -= 1


def _assert_balanced_anchors(chunk):
    checker = _AnchorBalanceChecker()
    checker.feed(chunk)
    checker.close()
    assert checker.open_anchors == 0, "chunk에 열린 <a> 태그가 남아있음: {!r}".format(chunk[:120])


# ─────────────────────────────────────────────
# format_briefing_html — html.escape
# ─────────────────────────────────────────────

class TestFormatBriefingHtmlEscaping:
    def test_escapes_script_tag_in_sentence_text(self):
        draft = _draft(sentences=[
            {"idx": 0, "section": "nasdaq",
             "text": "<script>alert(1)</script> 위험 문구", "article_ids": ["market_data"]},
        ])
        out = format_briefing_html(draft, 1, ARTICLES_BY_ID, 0.9, 0.8)

        assert "<script>alert(1)</script>" not in out
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out

    def test_escapes_ampersand_in_rationale(self):
        draft = _draft(forecasts=[
            {"market": "KOSPI", "direction": "up", "rationale": "AI & 반도체 강세"},
            {"market": "KOSDAQ", "direction": "flat", "rationale": "보합"},
        ])
        out = format_briefing_html(draft, 1, ARTICLES_BY_ID, 0.9, 0.8)

        assert "AI & 반도체" not in out
        assert "AI &amp; 반도체" in out

    def test_escapes_quote_injection_in_article_url(self):
        malicious_articles = {
            1: {"url": 'https://evil.com/"><script>alert(1)</script>', "title": "x"},
            2: {"url": "https://cnbc.com/b", "title": "연준 발언"},
        }
        draft = _draft()
        out = format_briefing_html(draft, 1, malicious_articles, 0.9, 0.8)

        assert '"><script>' not in out
        assert "&quot;" in out


# ─────────────────────────────────────────────
# format_briefing_html — 인용 링크
# ─────────────────────────────────────────────

class TestCitationLinks:
    def test_link_count_and_href_match_article_ids(self):
        draft = _draft(sentences=[
            {"idx": 0, "section": "cause", "text": "복수 인용 문장",
             "article_ids": [1, 2]},
        ])
        out = format_briefing_html(draft, 1, ARTICLES_BY_ID, 0.9, 0.8)

        assert out.count("<a ") == 2
        assert '<a href="https://reuters.com/a">[1]</a>' in out
        assert '<a href="https://cnbc.com/b">[2]</a>' in out

    def test_market_data_tag_has_no_link(self):
        draft = _draft(sentences=[
            {"idx": 0, "section": "nasdaq", "text": "지수 서술", "article_ids": ["market_data"]},
        ])
        out = format_briefing_html(draft, 1, ARTICLES_BY_ID, 0.9, 0.8)

        assert "<a " not in out
        assert "market_data" not in out

    def test_missing_article_mapping_is_skipped_without_crash(self):
        draft = _draft(sentences=[
            {"idx": 0, "section": "cause", "text": "매핑 없는 인용", "article_ids": [999]},
        ])
        out = format_briefing_html(draft, 1, ARTICLES_BY_ID, 0.9, 0.8)

        assert "<a " not in out
        assert "매핑 없는 인용" in out


# ─────────────────────────────────────────────
# format_briefing_html — warn_idxs / 신뢰 점수 배지 / 전망
# ─────────────────────────────────────────────

class TestFormatBriefingHtmlContent:
    def test_warn_idxs_prefix_shown(self):
        out = format_briefing_html(_draft(), 1, ARTICLES_BY_ID, 0.9, 0.8, warn_idxs=[1])

        lines = out.split("\n")
        warned = [l for l in lines if "기술주 강세가 상승을 이끌었다" in l][0]
        not_warned = [l for l in lines if "나스닥은 1.5% 상승" in l][0]

        assert warned.startswith("⚠️ ")
        assert not not_warned.startswith("⚠️ ")

    def test_trust_score_badge_present(self):
        out = format_briefing_html(_draft(), 1, ARTICLES_BY_ID, 0.856, 0.734)

        assert "🔎 사실정확성 86% · 근거연결 73%" in out

    def test_forecast_direction_emoji(self):
        out = format_briefing_html(_draft(), 1, ARTICLES_BY_ID, 0.9, 0.8)

        assert "📈" in out  # up
        assert "➡️" in out  # flat

    def test_section_titles_present(self):
        out = format_briefing_html(_draft(), 1, ARTICLES_BY_ID, 0.9, 0.8)

        assert "🇺🇸 나스닥 시황" in out
        assert "📊 등락 원인" in out
        assert "🇰🇷 한국 증시 영향" in out
        assert "💼 보유종목 뉴스" in out


# ─────────────────────────────────────────────
# split_html_by_length — 태그 경계 분할
# ─────────────────────────────────────────────

class TestSplitHtmlByLength:
    def test_short_text_returns_single_chunk(self):
        chunks = split_html_by_length("짧은 문장", max_len=4096)
        assert chunks == ["짧은 문장"]

    def test_all_chunks_within_max_len(self):
        html_text = "가나다라 " * 2000
        chunks = split_html_by_length(html_text, max_len=100)
        assert all(len(c) <= 100 for c in chunks)

    def test_reconstruction_preserves_content(self):
        html_text = "AAAA BBBB CCCC DDDD " * 50
        chunks = split_html_by_length(html_text, max_len=50)
        assert "".join(chunks) == html_text

    def test_anchor_tag_not_split_across_boundary(self):
        # 4096 경계 바로 앞에 긴 텍스트를 배치하고, 경계에 걸치도록 <a> 태그를 이어붙인다.
        prefix = "본문 " * 1400  # 4200자 — [:4090] 슬라이스가 실제로 4090자를 채우도록 보정(원래 1000배수는 3000자뿐이라 경계에 못 미쳤음)
        anchor = '<a href="https://reuters.com/very/long/path/article">[1]</a>'
        html_text = prefix[:4090] + anchor + "이어지는 본문 " * 10

        chunks = split_html_by_length(html_text, max_len=4096)

        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 4096
            _assert_balanced_anchors(chunk)

        # 앵커 전체가 한 조각 안에 온전히 들어있어야 한다 (분할되지 않음)
        assert any(anchor in chunk for chunk in chunks)
        assert "".join(chunks) == html_text

    def test_oversized_atomic_anchor_placed_alone(self):
        anchor = "<a href=\"https://example.com\">" + ("x" * 5000) + "</a>"
        chunks = split_html_by_length(anchor, max_len=100)

        assert chunks == [anchor]

    def test_empty_input_returns_single_empty_chunk(self):
        assert split_html_by_length("", max_len=4096) == [""]


# ─────────────────────────────────────────────
# build_feedback_keyboard_payload
# ─────────────────────────────────────────────

class TestFeedbackKeyboardPayload:
    def test_structure_and_callback_data(self):
        payload = build_feedback_keyboard_payload(42)

        assert payload == [[
            {"text": "👍", "callback_data": make_callback_data("up", 42)},
            {"text": "👎", "callback_data": make_callback_data("down", 42)},
        ]]

    def test_callback_data_roundtrips(self):
        from news_briefing.constants import parse_callback_data

        payload = build_feedback_keyboard_payload(7)
        up_data = payload[0][0]["callback_data"]
        down_data = payload[0][1]["callback_data"]

        assert parse_callback_data(up_data) == ("up", 7)
        assert parse_callback_data(down_data) == ("down", 7)
