from __future__ import annotations

"""
krx_universe.py - KRX 전체 종목 1차 스크리닝

FinanceDataReader.StockListing()으로 KOSPI+KOSDAQ 실시간 시장 데이터 조회 후
거래대금 상위 + 등락률 양수 종목만 추려 반환.
ML 스캔 대상을 전체 2000+ 종목에서 ~100종목으로 압축.

⚠️ GATE A — 생존편향(Survivorship Bias) 한계:
  get_krx_backtest_universe()는 실행 시점(현재)의 시가총액 기준으로 종목을 선정한다.
  즉, 2021~2024년 학습 데이터에 2026년 현재 살아남은 종목만 포함되므로
  상폐·합병·급락한 종목은 제외된다. 이는 백테스트 성과를 낙관적으로 편향시킨다.

  진정한 point-in-time universe는 각 거래일 t 기준의 지수 구성종목 이력 데이터가
  필요하나 FinanceDataReader/yfinance 가 이를 제공하지 않는다.

  대안: _STATIC_LONG_LISTED_UNIVERSE — 2010년 이전부터 KOSPI 상장된
  대형주 30종목 고정 리스트. 상폐 위험이 사실상 없으며 실질적 편향이 최소화됨.
  get_krx_backtest_universe(use_static=True) 로 활성화.
"""

import logging

logger = logging.getLogger(__name__)

# 2010년 이전 상장, 현재도 유효한 KOSPI 대형주 고정 리스트 (생존편향 최소화)
# 상폐·분할·합병 없이 연속 거래된 종목만 포함
_STATIC_LONG_LISTED_UNIVERSE: dict[str, str] = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "005380.KS": "현대차",
    "000270.KS": "기아",
    "051910.KS": "LG화학",
    "006400.KS": "삼성SDI",
    "035420.KS": "NAVER",
    "068270.KS": "셀트리온",
    "028260.KS": "삼성물산",
    "105560.KS": "KB금융",
    "055550.KS": "신한지주",
    "086790.KS": "하나금융지주",
    "316140.KS": "우리금융지주",
    "012330.KS": "현대모비스",
    "096770.KS": "SK이노베이션",
    "034730.KS": "SK",
    "017670.KS": "SK텔레콤",
    "030200.KS": "KT",
    "003550.KS": "LG",
    "066570.KS": "LG전자",
    "009540.KS": "HD한국조선해양",
    "010130.KS": "고려아연",
    "004020.KS": "현대제철",
    "011170.KS": "롯데케미칼",
    "021240.KS": "코웨이",
    "000810.KS": "삼성화재",
    "032830.KS": "삼성생명",
    "009830.KS": "한화솔루션",
    "042660.KS": "한화오션",
    "003490.KS": "대한항공",
}


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


def get_krx_backtest_universe(top_n: int = 200, use_static: bool = False) -> dict[str, str]:
    """
    백테스트용 KRX 유니버스.

    ⚠️ 기본 모드(use_static=False): 실행 시점 시가총액 상위 top_n 종목.
       생존편향 있음 — 현재 살아남은 종목만 포함.

    use_static=True: _STATIC_LONG_LISTED_UNIVERSE (30종목, 2010년 이전 상장 고정).
       생존편향이 사실상 없으나 종목 수가 적고 소형주 미포함.

    GATE A 검증 방법: use_static=True 로 재실행 후 EV가 여전히 > 0인지 확인.
    """
    if use_static:
        logger.info("백테스트 유니버스: 정적 장기상장 30종목 (생존편향 최소화)")
        return dict(_STATIC_LONG_LISTED_UNIVERSE)

    try:
        import FinanceDataReader as fdr
    except ImportError:
        logger.warning("FinanceDataReader 미설치")
        return {}

    candidates: dict[str, str] = {}

    for market, suffix in [("KOSPI", ".KS"), ("KOSDAQ", ".KQ")]:
        try:
            df = fdr.StockListing(market)
            if df is None or df.empty:
                logger.warning("%s 데이터 없음", market)
                continue

            # 우선주·스팩 제외 (종목코드가 0으로 끝나지 않으면 우선주)
            df = df[df["Code"].str.endswith("0")].copy()

            if "Marcap" in df.columns:
                df = df[df["Marcap"] > 0].nlargest(top_n // 2, "Marcap")
            else:
                df = df.head(top_n // 2)

            for _, row in df.iterrows():
                candidates[f"{row['Code']}{suffix}"] = row["Name"]

            logger.info("%s 백테스트 후보: %d개", market, len(df))
        except Exception as e:
            logger.warning("KRX %s 조회 실패: %s", market, e)

    logger.info("KRX 백테스트 유니버스: %d개", len(candidates))
    return candidates
