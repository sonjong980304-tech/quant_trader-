from __future__ import annotations

"""
krx_universe.py - KRX 전체 종목 1차 스크리닝

pykrx로 KOSPI+KOSDAQ 오늘 시장 데이터 조회 후
거래대금 상위 + 등락률 양수 종목만 추려 반환.
ML 스캔 대상을 전체 2000+ 종목에서 ~100종목으로 압축.
"""

import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)


def _latest_trading_date() -> str:
    """최근 영업일 (오늘 포함, 주말 건너뜀)"""
    d = date.today()
    for _ in range(7):
        if d.weekday() < 5:
            return d.strftime("%Y%m%d")
        d -= timedelta(days=1)
    return date.today().strftime("%Y%m%d")


def get_krx_candidates(top_n: int = 100) -> dict[str, str]:
    """
    오늘 시장 기준 KRX 후보 종목 반환.

    필터 기준:
      1. 거래량 > 0
      2. 등락률 > 0% (상승 중인 종목)
      3. KOSPI / KOSDAQ 각각 거래대금 상위 top_n/2개

    반환: {ticker: name}  예) {"005930.KS": "삼성전자"}
    """
    try:
        from pykrx import stock as krx
    except ImportError:
        logger.warning("pykrx 미설치 — pip install pykrx")
        return {}

    date_str   = _latest_trading_date()
    candidates: dict[str, str] = {}

    for market, suffix in [("KOSPI", ".KS"), ("KOSDAQ", ".KQ")]:
        try:
            df = krx.get_market_ohlcv_by_ticker(date_str, market=market)
            if df is None or df.empty:
                logger.warning("%s 데이터 없음 (%s)", market, date_str)
                continue

            # 거래량 있고 등락률 양수인 종목만
            df = df[df["거래량"] > 0]
            if "등락률" in df.columns:
                df = df[df["등락률"] > 0]

            # 거래대금 기준 상위 top_n/2개
            col = "거래대금" if "거래대금" in df.columns else df.columns[-1]
            top = df.nlargest(top_n // 2, col)

            for ticker in top.index:
                try:
                    name = krx.get_market_ticker_name(str(ticker))
                except Exception:
                    name = str(ticker)
                candidates[f"{ticker}{suffix}"] = name

            logger.info("%s 후보: %d개", market, len(top))

        except Exception as e:
            logger.warning("KRX %s 조회 실패: %s", market, e)

    logger.info("KRX 전체 후보 종목: %d개", len(candidates))
    return candidates
