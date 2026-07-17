"""
news_briefing/positions.py — 보유 종목 조회 (페이퍼 트레이딩 + KIS 실계좌 합집합)
"""

import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PAPER_POSITIONS_PATH = os.path.join(_PROJECT_ROOT, "paper_positions.json")


def _strip_market_suffix(ticker: str) -> str:
    """티커에서 시장 접미사(.KS/.KQ)를 제거해 KIS 잔고(접미사 없음)와 매칭 가능한 코드로 변환."""
    return ticker.replace(".KS", "").replace(".KQ", "")


def _get_paper_holdings() -> dict:
    """paper_positions.json에서 {ticker: name} 추출. 파일 없음/파싱 실패 시 빈 dict."""
    try:
        with open(_PAPER_POSITIONS_PATH, encoding="utf-8") as f:
            positions = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("[Positions] paper_positions.json 로드 실패: %s", e)
        return {}

    holdings = {}
    if isinstance(positions, dict):
        for v in positions.values():
            ticker = v.get("ticker")
            name = v.get("name")
            if ticker:
                holdings[ticker] = name
    return holdings


def _get_kis_holdings() -> dict:
    """
    KIS 실계좌 잔고 조회 → {stock_code: name} (stock_code는 시장 접미사 없음).
    dashboard/kis_live.py의 _get_trader()를 재사용한다.
    네트워크·키 실패 시 경고 로그 후 빈 dict 반환(예외 전파 금지).
    """
    try:
        dashboard_dir = os.path.join(_PROJECT_ROOT, "dashboard")
        if dashboard_dir not in sys.path:
            sys.path.insert(0, dashboard_dir)
        import kis_live

        trader = kis_live._get_trader()
        if trader is None:
            logger.warning("[Positions] KIS 라이브 조회 비활성(키 없음 또는 초기화 실패)")
            return {}
        balance = trader.get_balance()
    except Exception as e:
        logger.warning("[Positions] KIS 잔고 조회 실패: %s", e)
        return {}

    return {
        item["stock_code"]: item.get("name")
        for item in balance
        if item.get("stock_code")
    }


def get_holdings() -> list:
    """
    보유 종목 합집합 반환: 페이퍼 트레이딩 + KIS 실계좌.
    동일 종목(코드 기준)이 양쪽에 있으면 source="both", 한쪽에만 있으면 "paper"/"kis".
    반환: [{"ticker": str, "name": str, "source": "paper"|"kis"|"both"}, ...]
    """
    paper = _get_paper_holdings()
    kis = _get_kis_holdings()

    paper_codes = {_strip_market_suffix(ticker): ticker for ticker in paper}

    result = []
    seen_codes = set()

    for code, ticker in paper_codes.items():
        seen_codes.add(code)
        source = "both" if code in kis else "paper"
        result.append({"ticker": ticker, "name": paper[ticker], "source": source})

    for code, name in kis.items():
        if code in seen_codes:
            continue
        result.append({"ticker": code, "name": name, "source": "kis"})

    return result
