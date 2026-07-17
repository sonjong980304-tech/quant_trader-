"""
tests/test_nb_sources.py — news_briefing.sources 검증

feedparser / requests 전부 mock 처리한다(실제 네트워크 호출 없음).
실측 근거: .omc/research/nb-recon-20260710.md (해외 RSS는 CNBC 카테고리 4개만 실사용
가능, 네이버는 sort=sim + originallink 기준 도메인 필터 필요).
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from news_briefing import sources  # noqa: E402


# ─────────────────────────────────────────────
# fetch_rss
# ─────────────────────────────────────────────

def _mock_entry(title, summary, link, published):
    """feedparser entry는 dict-like(FeedParserDict) 객체이므로 dict로 충분히 흉내낼 수 있다."""
    return {"title": title, "summary": summary, "link": link, "published": published}


class _MockParsedFeed:
    def __init__(self, entries, bozo=0):
        self.entries = entries
        self.bozo = bozo


class TestFetchRss:
    def test_parses_all_whitelisted_feeds(self, monkeypatch):
        """4개 CNBC 피드 전부 정상 파싱되면 기사 dict가 계약대로 채워져야 한다."""

        def _fake_parse(url):
            return _MockParsedFeed([
                _mock_entry(
                    "Markets rally on rate cut hopes",
                    "Stocks rose broadly today.",
                    "https://www.cnbc.com/2026/07/10/markets-rally.html",
                    "Fri, 10 Jul 2026 09:00:00 GMT",
                )
            ])

        mock_feedparser = types.ModuleType("feedparser")
        mock_feedparser.parse = _fake_parse
        monkeypatch.setattr(sources, "feedparser", mock_feedparser)

        articles, failed_sources = sources.fetch_rss()

        assert failed_sources == []
        assert len(articles) == len(sources.FOREIGN_RSS_FEEDS)
        for art in articles:
            assert art["url"] == "https://www.cnbc.com/2026/07/10/markets-rally.html"
            assert art["domain"] == "www.cnbc.com"
            assert art["source_lang"] == "en"
            assert art["title"] == "Markets rally on rate cut hopes"
            assert art["body"] == "Stocks rose broadly today."
            assert art["published_at"] == "Fri, 10 Jul 2026 09:00:00 GMT"
            assert art["http_status"] is None

    def test_partial_feed_failure_isolated(self, monkeypatch):
        """한 피드가 예외를 던져도 나머지 피드는 계속 처리되고, 실패 소스명만 기록된다."""
        failing_url = sources.FOREIGN_RSS_FEEDS["cnbc_markets"]

        def _fake_parse(url):
            if url == failing_url:
                raise ValueError("network down")
            return _MockParsedFeed([
                _mock_entry(
                    "Fed holds rates steady",
                    "The Fed kept rates unchanged.",
                    "https://www.cnbc.com/2026/07/10/fed-holds.html",
                    "Fri, 10 Jul 2026 10:00:00 GMT",
                )
            ])

        mock_feedparser = types.ModuleType("feedparser")
        mock_feedparser.parse = _fake_parse
        monkeypatch.setattr(sources, "feedparser", mock_feedparser)

        articles, failed_sources = sources.fetch_rss()

        assert failed_sources == ["cnbc_markets"]
        # 나머지 3개 피드는 각 1건씩 정상 수집
        assert len(articles) == len(sources.FOREIGN_RSS_FEEDS) - 1

    def test_all_feeds_fail_returns_empty_articles(self, monkeypatch):
        def _fake_parse(url):
            raise ConnectionError("dns failure")

        mock_feedparser = types.ModuleType("feedparser")
        mock_feedparser.parse = _fake_parse
        monkeypatch.setattr(sources, "feedparser", mock_feedparser)

        articles, failed_sources = sources.fetch_rss()

        assert articles == []
        assert sorted(failed_sources) == sorted(sources.FOREIGN_RSS_FEEDS.keys())

    def test_off_whitelist_domain_entry_is_dropped(self, monkeypatch):
        """만에 하나 피드가 화이트리스트 밖 도메인 링크를 반환하면 방어적으로 제외한다."""

        def _fake_parse(url):
            return _MockParsedFeed([
                _mock_entry(
                    "Suspicious redirect",
                    "Not from cnbc.",
                    "https://notcnbc.com/article.html",
                    "Fri, 10 Jul 2026 09:00:00 GMT",
                )
            ])

        mock_feedparser = types.ModuleType("feedparser")
        mock_feedparser.parse = _fake_parse
        monkeypatch.setattr(sources, "feedparser", mock_feedparser)

        articles, failed_sources = sources.fetch_rss()

        assert failed_sources == []
        assert articles == []


# ─────────────────────────────────────────────
# fetch_naver
# ─────────────────────────────────────────────

class _MockResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP error {}".format(self.status_code))

    def json(self):
        return self._json_data


class TestFetchNaver:
    def test_filters_by_originallink_domain(self, monkeypatch):
        """item['link']는 항상 n.news.naver.com이라 무의미 — originallink 기준으로 판정해야 한다."""
        items = [
            {
                "title": "삼성전자 주가 상승",
                "link": "https://n.news.naver.com/mnews/article/001/000001",
                "originallink": "https://www.yna.co.kr/view/AKR20260710000100",
                "description": "삼성전자 주가가 상승했다.",
                "pubDate": "Fri, 10 Jul 2026 09:00:00 +0900",
            },
            {
                "title": "삼성전자 관련 잡음",
                "link": "https://n.news.naver.com/mnews/article/002/000002",
                "originallink": "https://www.tokenpost.kr/article/999",
                "description": "화이트리스트 밖 매체.",
                "pubDate": "Fri, 10 Jul 2026 09:05:00 +0900",
            },
            {
                "title": "매경 서브도메인",
                "link": "https://n.news.naver.com/mnews/article/003/000003",
                "originallink": "https://mbn.mk.co.kr/news/economy/12345",
                "description": "mk.co.kr 서브도메인도 허용.",
                "pubDate": "Fri, 10 Jul 2026 09:10:00 +0900",
            },
            {
                "title": "originallink 없음",
                "link": "https://n.news.naver.com/mnews/article/004/000004",
                "originallink": "",
                "description": "네이버 자체 컨텐츠.",
                "pubDate": "Fri, 10 Jul 2026 09:15:00 +0900",
            },
        ]

        captured = {}

        def _fake_get(url, headers=None, params=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["params"] = params
            captured["timeout"] = timeout
            return _MockResponse({"items": items})

        mock_requests = types.ModuleType("requests")
        mock_requests.get = _fake_get
        monkeypatch.setattr(sources, "requests", mock_requests)

        articles = sources.fetch_naver("삼성전자 주식", client_id="cid", client_secret="secret")

        assert len(articles) == 2
        domains = {a["domain"] for a in articles}
        assert domains == {"www.yna.co.kr", "mbn.mk.co.kr"}
        for art in articles:
            assert art["source_lang"] == "ko"
            assert art["url"] != "https://n.news.naver.com/mnews/article/001/000001"
            assert art["http_status"] is None

    def test_uses_sim_sort_and_display_30(self, monkeypatch):
        captured = {}

        def _fake_get(url, headers=None, params=None, timeout=None):
            captured["params"] = params
            captured["headers"] = headers
            return _MockResponse({"items": []})

        mock_requests = types.ModuleType("requests")
        mock_requests.get = _fake_get
        monkeypatch.setattr(sources, "requests", mock_requests)

        sources.fetch_naver("코스피", client_id="cid", client_secret="secret")

        assert captured["params"]["sort"] == "sim"
        assert captured["params"]["display"] == 30
        assert captured["params"]["query"] == "코스피"
        assert captured["headers"]["X-Naver-Client-Id"] == "cid"
        assert captured["headers"]["X-Naver-Client-Secret"] == "secret"

    def test_missing_credentials_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(sources, "NAVER_CLIENT_ID", "")
        monkeypatch.setattr(sources, "NAVER_CLIENT_SECRET", "")

        articles = sources.fetch_naver("삼성전자 주식")

        assert articles == []

    def test_injected_credentials_override_config(self, monkeypatch):
        """client_id/client_secret 인자가 주어지면 config 값 대신 그것을 쓴다(테스트용 주입)."""
        monkeypatch.setattr(sources, "NAVER_CLIENT_ID", "")
        monkeypatch.setattr(sources, "NAVER_CLIENT_SECRET", "")

        captured = {}

        def _fake_get(url, headers=None, params=None, timeout=None):
            captured["headers"] = headers
            return _MockResponse({"items": []})

        mock_requests = types.ModuleType("requests")
        mock_requests.get = _fake_get
        monkeypatch.setattr(sources, "requests", mock_requests)

        sources.fetch_naver("삼성전자 주식", client_id="injected_id", client_secret="injected_secret")

        assert captured["headers"]["X-Naver-Client-Id"] == "injected_id"
        assert captured["headers"]["X-Naver-Client-Secret"] == "injected_secret"

    def test_request_exception_returns_empty_list(self, monkeypatch):
        def _fake_get(url, headers=None, params=None, timeout=None):
            raise ConnectionError("network down")

        mock_requests = types.ModuleType("requests")
        mock_requests.get = _fake_get
        monkeypatch.setattr(sources, "requests", mock_requests)

        articles = sources.fetch_naver("삼성전자 주식", client_id="cid", client_secret="secret")

        assert articles == []


# ─────────────────────────────────────────────
# fetch_fulltext
# ─────────────────────────────────────────────

class TestFetchFulltext:
    def _base_article(self):
        return {
            "url": "https://www.cnbc.com/2026/07/10/some-article.html",
            "domain": "www.cnbc.com",
            "source_lang": "en",
            "title": "Some article",
            "body": "Snippet body from RSS summary.",
            "published_at": "Fri, 10 Jul 2026 09:00:00 GMT",
            "http_status": None,
        }

    def test_success_replaces_body_and_records_status(self, monkeypatch):
        html = (
            "<html><body>"
            "<p>SUBSCRIBE TO CNBC NEWSLETTER</p>"
            "<p>" + ("First real paragraph with real article content padded out. " * 2) + "</p>"
            "<p>" + ("Second real paragraph continuing the article body text here. " * 2) + "</p>"
            "</body></html>"
        )

        def _fake_get(url, timeout=None, headers=None):
            assert "User-Agent" in headers
            resp = _MockResponse({}, status_code=200)
            resp.text = html
            return resp

        mock_requests = types.ModuleType("requests")
        mock_requests.get = _fake_get
        monkeypatch.setattr(sources, "requests", mock_requests)

        article = self._base_article()
        result = sources.fetch_fulltext(article, timeout=5)

        assert result["http_status"] == 200
        assert "First real paragraph" in result["body"]
        assert "Second real paragraph" in result["body"]
        # 짧은 노이즈 문단(뉴스레터 안내)은 제외되어야 한다
        assert "SUBSCRIBE TO CNBC NEWSLETTER" not in result["body"]

    def test_timeout_falls_back_to_snippet(self, monkeypatch):
        import requests as real_requests

        def _fake_get(url, timeout=None, headers=None):
            raise real_requests.exceptions.Timeout("timed out")

        mock_requests = types.ModuleType("requests")
        mock_requests.get = _fake_get
        mock_requests.exceptions = real_requests.exceptions
        monkeypatch.setattr(sources, "requests", mock_requests)

        article = self._base_article()
        original_body = article["body"]
        result = sources.fetch_fulltext(article, timeout=5)

        assert result["body"] == original_body
        assert result["http_status"] is None

    def test_4xx_falls_back_to_snippet_and_records_status(self, monkeypatch):
        def _fake_get(url, timeout=None, headers=None):
            resp = _MockResponse({}, status_code=404)
            resp.text = "<html><body>Not Found</body></html>"
            return resp

        mock_requests = types.ModuleType("requests")
        mock_requests.get = _fake_get
        monkeypatch.setattr(sources, "requests", mock_requests)

        article = self._base_article()
        original_body = article["body"]
        result = sources.fetch_fulltext(article, timeout=5)

        assert result["body"] == original_body
        assert result["http_status"] == 404

    def test_no_paragraphs_falls_back_to_snippet(self, monkeypatch):
        def _fake_get(url, timeout=None, headers=None):
            resp = _MockResponse({}, status_code=200)
            resp.text = "<html><body>Oops, something went wrong</body></html>"
            return resp

        mock_requests = types.ModuleType("requests")
        mock_requests.get = _fake_get
        monkeypatch.setattr(sources, "requests", mock_requests)

        article = self._base_article()
        original_body = article["body"]
        result = sources.fetch_fulltext(article, timeout=5)

        assert result["body"] == original_body
        assert result["http_status"] == 200

    def test_exception_never_propagates(self, monkeypatch):
        def _fake_get(url, timeout=None, headers=None):
            raise RuntimeError("unexpected boom")

        mock_requests = types.ModuleType("requests")
        mock_requests.get = _fake_get
        monkeypatch.setattr(sources, "requests", mock_requests)

        article = self._base_article()
        # 예외가 전파되지 않고 원래 article이 반환되어야 한다
        result = sources.fetch_fulltext(article, timeout=5)
        assert result["body"] == article["body"]


# ─────────────────────────────────────────────
# fetch_fulltext_batch
# ─────────────────────────────────────────────

class TestFetchFulltextBatch:
    def test_enforces_limit_of_12(self, monkeypatch):
        calls = []

        def _fake_get(url, timeout=None, headers=None):
            calls.append(url)
            resp = _MockResponse({}, status_code=200)
            resp.text = "<html><body><p>" + ("Real long enough paragraph body text here. " * 3) + "</p></body></html>"
            return resp

        mock_requests = types.ModuleType("requests")
        mock_requests.get = _fake_get
        monkeypatch.setattr(sources, "requests", mock_requests)

        articles = [
            {
                "url": "https://www.cnbc.com/article-{}.html".format(i),
                "domain": "www.cnbc.com",
                "source_lang": "en",
                "title": "Article {}".format(i),
                "body": "Snippet {}".format(i),
                "published_at": "Fri, 10 Jul 2026 09:00:00 GMT",
                "http_status": None,
            }
            for i in range(20)
        ]

        result = sources.fetch_fulltext_batch(articles, limit=12, timeout=5)

        assert len(result) == 20
        assert len(calls) == 12
        # 앞 12건은 http_status가 채워짐(fetch 시도됨)
        for art in result[:12]:
            assert art["http_status"] == 200
        # 나머지는 스니펫 그대로, http_status는 None(미시도)
        for i, art in enumerate(result[12:], start=12):
            assert art["http_status"] is None
            assert art["body"] == "Snippet {}".format(i)

    def test_default_limit_is_12(self, monkeypatch):
        calls = []

        def _fake_get(url, timeout=None, headers=None):
            calls.append(url)
            resp = _MockResponse({}, status_code=200)
            resp.text = "<html><body><p>" + ("Real long enough paragraph body text here. " * 3) + "</p></body></html>"
            return resp

        mock_requests = types.ModuleType("requests")
        mock_requests.get = _fake_get
        monkeypatch.setattr(sources, "requests", mock_requests)

        articles = [
            {
                "url": "https://www.cnbc.com/article-{}.html".format(i),
                "domain": "www.cnbc.com",
                "source_lang": "en",
                "title": "Article {}".format(i),
                "body": "Snippet {}".format(i),
                "published_at": "Fri, 10 Jul 2026 09:00:00 GMT",
                "http_status": None,
            }
            for i in range(15)
        ]

        result = sources.fetch_fulltext_batch(articles)

        assert len(calls) == 12

    def test_fewer_than_limit_articles_all_fetched(self, monkeypatch):
        calls = []

        def _fake_get(url, timeout=None, headers=None):
            calls.append(url)
            resp = _MockResponse({}, status_code=200)
            resp.text = "<html><body><p>" + ("Real long enough paragraph body text here. " * 3) + "</p></body></html>"
            return resp

        mock_requests = types.ModuleType("requests")
        mock_requests.get = _fake_get
        monkeypatch.setattr(sources, "requests", mock_requests)

        articles = [
            {
                "url": "https://www.cnbc.com/article-{}.html".format(i),
                "domain": "www.cnbc.com",
                "source_lang": "en",
                "title": "Article {}".format(i),
                "body": "Snippet {}".format(i),
                "published_at": "Fri, 10 Jul 2026 09:00:00 GMT",
                "http_status": None,
            }
            for i in range(5)
        ]

        result = sources.fetch_fulltext_batch(articles, limit=12, timeout=5)

        assert len(calls) == 5
        assert len(result) == 5
