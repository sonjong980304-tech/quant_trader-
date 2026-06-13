"""
market_calendar.py - KRX 영업일 체크 유틸리티

pykrx.get_market_ohlcv_by_date()로 연간 영업일 목록을 캐시해
주말 + 공휴일(빨간날) 모두 자동 필터링.
"""

import logging
from datetime import datetime, date
import pytz

logger = logging.getLogger(__name__)
KST = pytz.timezone("Asia/Seoul")

_CACHE: set = set()
_CACHE_YEAR: int = 0
_CACHE_DATE: "date | None" = None   # 캐시를 마지막으로 갱신한 날짜 (일 단위 재갱신용)


def is_kr_trading_day(d: date = None) -> bool:
    """
    해당 날짜가 KRX 영업일인지 확인 (일별 캐시).
    d=None이면 오늘(KST) 기준.
    실패 시 주말 체크로 폴백.

    주의: pykrx OHLCV 조회는 과거 거래일만 반환한다.
    오늘이 아직 캐시에 없으면(장 시작 전) 주말 여부로 폴백한다.
    """
    global _CACHE, _CACHE_YEAR, _CACHE_DATE
    if d is None:
        d = datetime.now(KST).date()
    year = d.year
    today = datetime.now(KST).date()

    # 연도 변경 또는 날짜 변경 시 캐시 재갱신
    if year != _CACHE_YEAR or _CACHE_DATE != today:
        try:
            from pykrx import stock as krx
            df = krx.get_market_ohlcv_by_date(f"{year}0101", f"{year}1231", "005930")
            _CACHE = {ts.date() for ts in df.index}
            _CACHE_YEAR = year
            _CACHE_DATE = today
            logger.info("KRX 영업일 캐시 갱신: %d년 %d거래일", year, len(_CACHE))
        except Exception as e:
            logger.warning("KRX 영업일 조회 실패 — 주말 체크로 폴백: %s", e)
            return d.weekday() < 5

    # 오늘 이후(미래 포함) 날짜가 캐시에 없으면 pykrx가 미래 데이터를 반환 안 하므로
    # 주말 체크로 폴백 (공휴일은 미반영되나 미래 날짜 스케줄링에는 충분)
    if d >= today and d not in _CACHE:
        return d.weekday() < 5

    return d in _CACHE
