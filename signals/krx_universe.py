from __future__ import annotations

"""
krx_universe.py - KRX 전체 종목 1차 스크리닝

FinanceDataReader.StockListing()으로 KOSPI+KOSDAQ 실시간 시장 데이터 조회 후
거래대금 상위 + 등락률 양수 종목만 추려 반환.
ML 스캔 대상을 전체 2000+ 종목에서 ~100종목으로 압축.
"""

import logging

logger = logging.getLogger(__name__)


def get_krx_candidates(top_n: int = 100) -> dict[str, str]:
    """
    현재 시장 기준 KRX 후보 종목 반환.

    필터 기준:
      1. 거래량 > 0
      2. 등락률 > 0% (상승 중인 종목)
      3. KOSPI / KOSDAQ 각각 거래대금 상위 top_n/2개

    반환: {ticker: name}  예) {"005930.KS": "삼성전자"}
    """
    try:
        import FinanceDataReader as fdr
    except ImportError:
        logger.warning("FinanceDataReader 미설치 — pip install finance-datareader")
        return {}

    candidates: dict[str, str] = {}

    for market, suffix in [("KOSPI", ".KS"), ("KOSDAQ", ".KQ")]:
        try:
            df = fdr.StockListing(market)
            if df is None or df.empty:
                logger.warning("%s 데이터 없음", market)
                continue

            df = df[df["Volume"] > 0].copy()
            if "ChagesRatio" in df.columns:
                df = df[df["ChagesRatio"] > 0]

            top = df.nlargest(top_n // 2, "Amount")

            for _, row in top.iterrows():
                ticker = f"{row['Code']}{suffix}"
                candidates[ticker] = row["Name"]

            logger.info("%s 후보: %d개", market, len(top))

        except Exception as e:
            logger.warning("KRX %s 조회 실패: %s", market, e)

    logger.info("KRX 전체 후보 종목: %d개", len(candidates))
    return candidates
