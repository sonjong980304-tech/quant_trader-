"""
news_fetcher.py - 네이버 뉴스 수집
"""

import re
import logging
import requests

from config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET

logger = logging.getLogger(__name__)

_NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def fetch_naver_news(stock_name: str, n: int = 3) -> list:
    """
    네이버 뉴스 API로 종목 관련 최신 뉴스 n건 반환.
    반환: [{"title": ..., "link": ..., "pubDate": ...}, ...]
    """
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        logger.warning("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 미설정 — 뉴스 수집 건너뜀")
        return []

    try:
        resp = requests.get(
            _NAVER_NEWS_URL,
            headers={
                "X-Naver-Client-Id":     NAVER_CLIENT_ID,
                "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
            },
            params={"query": f"{stock_name} 주식", "display": n, "sort": "date"},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])[:n]
        for item in items:
            item["title"]       = _strip_html(item.get("title", ""))
            item["description"] = _strip_html(item.get("description", ""))
        return items
    except Exception as e:
        logger.error("네이버 뉴스 수집 실패 (%s): %s", stock_name, e)
        return []
