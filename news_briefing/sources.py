"""
news_briefing/sources.py — 뉴스 직수집 (해외 RSS + 국내 네이버 뉴스 API + 원문 풀-페치)

화이트리스트는 실측 문서(.omc/research/nb-recon-20260710.md) 결과를 그대로 이식한 것이다.
- 해외 RSS: Reuters/AP는 공개 RSS 폐지(404/DNS 실패), MarketWatch/Yahoo는 RSS summary가
  없고 풀-페치도 봇 차단(401)·클라이언트 렌더링으로 막혀 있어 전부 제외 — CNBC 카테고리
  피드 4개만 실사용 가능.
- 국내 네이버 API: sort=date는 화이트리스트 매치율이 0~1/30으로 극히 낮고, sort=sim이
  최대 8/30으로 유의미하게 높다. 도메인 판정은 반드시 item["originallink"] 기준이어야
  한다(item["link"]는 항상 n.news.naver.com이라 무의미).

feedparser/requests는 모듈 속성으로 참조한다(테스트에서 monkeypatch로 쉽게 교체하기 위함).
"""

import logging
import re

import feedparser
import requests
from bs4 import BeautifulSoup

from config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 화이트리스트 (실측 확정 — nb-recon-20260710.md §5)
# ─────────────────────────────────────────────

FOREIGN_RSS_FEEDS = {
    "cnbc_markets": "https://www.cnbc.com/id/15839069/device/rss/rss.html",
    "cnbc_finance": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "cnbc_economy": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "cnbc_world_markets": "https://www.cnbc.com/id/15839135/device/rss/rss.html",
}
FOREIGN_ALLOWED_DOMAINS = {"cnbc.com"}

NAVER_QUERY_SORT = "sim"      # date 아님 — 실측: date 매치율 0~1/30 vs sim 최대 8/30
NAVER_QUERY_DISPLAY = 30      # 필터 후 상위 N 뽑기 위해 넉넉히 조회
KR_ALLOWED_DOMAINS = {"yna.co.kr", "hankyung.com", "mk.co.kr", "biz.chosun.com"}

_NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"
_FETCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_MIN_PARAGRAPH_LEN = 80  # 이 미만은 네비게이션/뉴스레터 안내 등 노이즈로 간주해 제외


def _strip_html(text):
    # type: (str) -> str
    return re.sub(r"<[^>]+>", "", text).strip()


def _extract_domain(url):
    # type: (str) -> str
    """URL에서 호스트(netloc)를 소문자로 추출. 파싱 실패 시 빈 문자열."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _domain_allowed(domain, allowed_domains):
    # type: (str, set) -> bool
    """domain이 allowed_domains 중 하나와 정확히 일치하거나 그 서브도메인이면 True.
    (예: mbn.mk.co.kr → mk.co.kr 매치 인정, notcnbc.com → cnbc.com 오탐 방지)"""
    if not domain:
        return False
    for allowed in allowed_domains:
        if domain == allowed or domain.endswith("." + allowed):
            return True
    return False


def fetch_rss():
    """
    화이트리스트 해외 RSS(FOREIGN_RSS_FEEDS) 전부를 feedparser로 파싱한다.
    소스별로 try/except 격리 — 한 피드가 실패해도 나머지는 계속 처리된다.
    반환: (articles: list[dict], failed_sources: list[str])
    """
    articles = []
    failed_sources = []

    for name, url in FOREIGN_RSS_FEEDS.items():
        try:
            parsed = feedparser.parse(url)
            entries = getattr(parsed, "entries", [])
            if getattr(parsed, "bozo", 0) and not entries:
                raise ValueError("feed parse 실패 (bozo=1, entries 없음)")

            for entry in entries:
                link = entry.get("link", "")
                domain = _extract_domain(link)
                if not _domain_allowed(domain, FOREIGN_ALLOWED_DOMAINS):
                    continue
                articles.append({
                    "url": link,
                    "domain": domain,
                    "source_lang": "en",
                    "title": entry.get("title", "").strip(),
                    "body": entry.get("summary", "").strip(),
                    "published_at": entry.get("published", ""),
                    "http_status": None,
                })
        except Exception as e:
            logger.warning("[Sources] RSS 피드 수집 실패 (%s): %s", name, e)
            failed_sources.append(name)

    return articles, failed_sources


def fetch_naver(ticker_or_query, client_id=None, client_secret=None):
    """
    네이버 뉴스 API로 ticker_or_query 관련 기사를 조회한다(sort=sim, display=30).
    client_id/client_secret을 넘기면 config 값 대신 그것을 쓴다(테스트용 주입).
    item["originallink"] 도메인이 KR_ALLOWED_DOMAINS에 속하는 것만 통과시킨다
    (item["link"]는 항상 n.news.naver.com이라 도메인 판정에 쓸 수 없음).
    반환: articles: list[dict]
    """
    cid = client_id if client_id is not None else NAVER_CLIENT_ID
    csecret = client_secret if client_secret is not None else NAVER_CLIENT_SECRET

    if not cid or not csecret:
        logger.warning("[Sources] NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 미설정 — 국내 뉴스 수집 건너뜀")
        return []

    try:
        resp = requests.get(
            _NAVER_NEWS_URL,
            headers={
                "X-Naver-Client-Id": cid,
                "X-Naver-Client-Secret": csecret,
            },
            params={
                "query": ticker_or_query,
                "display": NAVER_QUERY_DISPLAY,
                "sort": NAVER_QUERY_SORT,
            },
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.error("[Sources] 네이버 뉴스 수집 실패 (%s): %s", ticker_or_query, e)
        return []

    articles = []
    for item in resp.json().get("items", []):
        original_link = item.get("originallink", "")
        if not original_link:
            continue
        domain = _extract_domain(original_link)
        if not _domain_allowed(domain, KR_ALLOWED_DOMAINS):
            continue
        articles.append({
            "url": original_link,
            "domain": domain,
            "source_lang": "ko",
            "title": _strip_html(item.get("title", "")),
            "body": _strip_html(item.get("description", "")),
            "published_at": item.get("pubDate", ""),
            "http_status": None,
        })

    return articles


def fetch_fulltext(article, timeout=5):
    """
    article["url"]로 원문을 GET해 <p> 태그를 집계, 본문 추출을 시도한다.
    성공 시 article["body"]를 원문으로 교체하고 article["http_status"]를 채운다.
    실패(타임아웃/4xx·5xx/본문 없음) 시 원래 body(스니펫)를 유지하되 http_status는
    시도한 경우 기록한다. 예외는 절대 전파하지 않는다(호출부 배치 처리 보호).
    article은 in-place로 갱신하고 그대로 반환한다.
    """
    try:
        resp = requests.get(
            article["url"],
            timeout=timeout,
            headers={"User-Agent": _FETCH_USER_AGENT},
        )
        article["http_status"] = resp.status_code
        if resp.status_code >= 400:
            return article

        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [_strip_html(str(p)) for p in soup.find_all("p")]
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) >= _MIN_PARAGRAPH_LEN]
        if paragraphs:
            article["body"] = "\n\n".join(paragraphs)
        return article
    except Exception as e:
        logger.warning("[Sources] 원문 풀-페치 실패 (%s): %s", article.get("url"), e)
        return article


def fetch_fulltext_batch(articles, limit=12, timeout=5):
    """
    articles 앞에서부터 최대 limit건에만 fetch_fulltext를 적용한다.
    나머지는 스니펫(원래 body) 그대로 두고 http_status도 건드리지 않는다(미시도 = None).
    반환: articles 전체(길이 동일, 앞 limit건만 in-place 갱신됨).
    """
    for article in articles[:limit]:
        fetch_fulltext(article, timeout=timeout)
    return articles
