"""
news_briefing/market_data.py — 시장 지수 스냅샷 조회 (미국·한국)

yfinance / FinanceDataReader 모두 함수 내부 지연 임포트한다.
FinanceDataReader는 테스트 venv(3.11)에 설치돼 있지 않으므로
모듈 최상단에서 임포트하면 `import news_briefing.market_data` 자체가 실패한다.
"""

import logging

logger = logging.getLogger(__name__)


def _change_pct(prev: float, last: float) -> float:
    """전일 대비 등락률(%) 계산. 소수점 2자리로 반올림한다 — 원시 float을 그대로 두면
    LLM 프롬프트·발송 메시지에 0.28511108421508896% 같은 과도한 정밀도가 그대로
    노출된다(오차 허용치 ±0.05%p보다 훨씬 작은 반올림이라 사실 검증에는 영향 없음)."""
    return round((last - prev) / prev * 100, 2)


def get_us_snapshot() -> dict:
    """
    나스닥종합(^IXIC)·S&P500(^GSPC)·다우존스(^DJI)·필라델피아반도체지수(^SOX)
    최근 2영업일 종가 및 등락률.
    반환: {"nasdaq": {...}, "sp500": {...}, "dow": {...}, "sox": {...}, "asof": "YYYY-MM-DD"}
    개별 지수 조회 실패는 해당 키만 빠지고 나머지는 정상 반환(부분 실패 허용).
    전부 조회 실패 시 빈 dict 반환.
    """
    import yfinance as yf

    tickers = {"nasdaq": "^IXIC", "sp500": "^GSPC", "dow": "^DJI", "sox": "^SOX"}
    result = {}
    asof = None
    for key, ticker in tickers.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) < 2:
                continue
            prev = float(hist["Close"].iloc[-2])
            last = float(hist["Close"].iloc[-1])
            result[key] = {"close": round(last, 2), "change_pct": _change_pct(prev, last)}
            asof = hist.index[-1].strftime("%Y-%m-%d")
        except Exception as e:
            logger.warning("[MarketData] %s 조회 실패: %s", ticker, e)

    if not result:
        return {}
    result["asof"] = asof
    return result


def get_kr_index_change(market: str):
    """
    KOSPI/KOSDAQ 최근 2영업일 종가 및 등락률.
    market: "KOSPI" 또는 "KOSDAQ"
    반환: {"close": float, "change_pct": float, "asof": "YYYY-MM-DD"}
    데이터 없거나 조회 실패 시 None(예외 아님 — 호출부 재시도 로직이 처리).
    """
    codes = {"KOSPI": "KS11", "KOSDAQ": "KQ11"}
    code = codes.get(market)
    if code is None:
        return None

    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(code)
    except Exception as e:
        logger.warning("[MarketData] %s 지수 조회 실패: %s", market, e)
        return None

    if df is None or len(df) < 2:
        return None

    tail = df.tail(2)
    prev = float(tail["Close"].iloc[-2])
    last = float(tail["Close"].iloc[-1])
    return {
        "close": round(last, 2),
        "change_pct": _change_pct(prev, last),
        "asof": tail.index[-1].strftime("%Y-%m-%d"),
    }


def get_kr_index_snapshot() -> dict:
    """KOSPI·KOSDAQ 스냅샷 묶음. 반환: {"kospi": {...}|None, "kosdaq": {...}|None}"""
    return {
        "kospi": get_kr_index_change("KOSPI"),
        "kosdaq": get_kr_index_change("KOSDAQ"),
    }
