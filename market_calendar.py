"""
market_calendar.py - KRX 영업일 체크 유틸리티

pykrx.get_trading_dates()로 연간 영업일 목록을 캐시해
주말 + 공휴일(빨간날) 모두 자동 필터링.
"""

import logging
from datetime import datetime, date
import pytz

logger = logging.getLogger(__name__)
KST = pytz.timezone("Asia/Seoul")

_CACHE: set = set()
_CACHE_YEAR: int = 0


def is_kr_trading_day(d: date = None) -> bool:
    """
    해당 날짜가 KRX 영업일인지 확인 (연간 캐시).
    d=None이면 오늘(KST) 기준.
    실패 시 주말 체크로 폴백.
    """
    global _CACHE, _CACHE_YEAR
    if d is None:
        d = datetime.now(KST).date()
    year = d.year
    if year != _CACHE_YEAR:
        try:
            from pykrx import stock as krx
            dates = krx.get_trading_dates(f"{year}0101", f"{year}1231")
            _CACHE = {dt.date() if hasattr(dt, "date") else dt for dt in dates}
            _CACHE_YEAR = year
            logger.info("KRX 영업일 캐시 갱신: %d년 %d거래일", year, len(_CACHE))
        except Exception as e:
            logger.warning("KRX 영업일 조회 실패 — 주말 체크로 폴백: %s", e)
            return d.weekday() < 5
    return d in _CACHE
