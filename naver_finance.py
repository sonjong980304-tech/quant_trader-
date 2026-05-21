"""
naver_finance.py - Naver 증권 재무정보 스크래핑

get_financials(identifier) 함수 하나로 종목명/코드를 받아
PER, PBR, EPS, BPS, 시가총액, 매출, 영업이익 등을 텍스트로 반환.
"""

import logging
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9",
}

# 우선 출력할 지표 순서
_PRIORITY = [
    "시가총액", "PER", "EPS", "추정PER", "추정EPS",
    "PBR", "BPS", "주당배당금", "시가배당률", "ROE",
    "매출액", "영업이익", "당기순이익", "부채비율", "유동비율",
]

# 이 키워드가 포함된 항목만 필터링
_KEEP_KEYWORDS = [
    "PER", "EPS", "PBR", "BPS", "ROE", "시가총액", "배당",
    "매출", "영업이익", "당기순이익", "부채", "유동비율",
    "상장주식", "외국인",
]


def _resolve_code(identifier: str) -> tuple[str, str]:
    """종목명 또는 티커 → (6자리코드, 종목명). 실패 시 ('', '')."""
    from config import STOCKS

    clean = identifier.replace(".KS", "").replace(".KQ", "").strip()

    for ticker, name in STOCKS.items():
        code = ticker.replace(".KS", "").replace(".KQ", "")
        if identifier.strip() in (ticker, code, name) or clean in (code, name):
            return code, name

    # identifier 자체가 6자리 숫자 코드인 경우
    if clean.isdigit() and len(clean) == 6:
        return clean, clean

    return "", ""


def _scrape_main(code: str) -> dict:
    """Naver Finance 메인 페이지 → 투자지표 딕셔너리 반환."""
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        resp.encoding = "utf-8"
    except Exception as e:
        logger.warning("Naver Finance 메인 조회 실패 (%s): %s", code, e)
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    result: dict[str, str] = {}

    # ── 방식 1: th + td 쌍 (수직 테이블) ──
    for row in soup.find_all("tr"):
        th = row.find("th")
        tds = row.find_all("td")
        if th and tds:
            key = th.get_text(strip=True)
            val = tds[0].get_text(strip=True)
            if key and val and val not in ("-", "N/A", ""):
                result.setdefault(key, val)

    # ── 방식 2: 헤더 행 + 데이터 행 (수평 테이블) ──
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if not headers:
            continue
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) == len(headers):
                for h, td in zip(headers, cells):
                    val = td.get_text(strip=True)
                    if h and val and val not in ("-", "N/A", ""):
                        result.setdefault(h, val)

    # ── 방식 3: dl/dt/dd (시가총액 등) ──
    for dl in soup.find_all("dl"):
        dt = dl.find("dt")
        dd = dl.find("dd")
        if dt and dd:
            key = dt.get_text(strip=True)
            val = dd.get_text(strip=True)
            if key and val:
                result.setdefault(key, val)

    return result


def _scrape_coinfo(code: str) -> dict:
    """Naver Finance 기업정보 페이지 → 연간 재무 요약 딕셔너리 반환."""
    url = f"https://finance.naver.com/item/coinfo.naver?code={code}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        resp.encoding = "utf-8"
    except Exception as e:
        logger.warning("Naver Finance coinfo 조회 실패 (%s): %s", code, e)
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    result: dict[str, str] = {}

    for row in soup.find_all("tr"):
        th = row.find("th")
        tds = row.find_all("td")
        if th and tds:
            key = th.get_text(strip=True)
            # 가장 최근 연도 값 (첫 번째 td)
            val = tds[0].get_text(strip=True)
            if key and val and val not in ("-", "N/A", ""):
                result.setdefault(key, val)

    return result


def get_financials(identifier: str) -> str:
    """
    종목명 또는 코드를 받아 Naver Finance 재무지표를 텍스트로 반환.
    STOCKS 목록에 없는 종목도 6자리 코드로 직접 조회 가능.
    """
    code, name = _resolve_code(identifier)
    if not code:
        return (
            f"'{identifier}' 종목을 찾을 수 없습니다.\n"
            "6자리 종목코드(예: 005930)나 정확한 종목명을 사용해주세요."
        )

    label = f"{name} ({code})" if name and name != code else code

    main_data  = _scrape_main(code)
    coinfo_data = _scrape_coinfo(code)
    merged = {**coinfo_data, **main_data}   # main_data 우선

    if not merged:
        return f"{label} 재무정보 조회 실패 — Naver Finance 연결 오류"

    lines = [f"[{label} — Naver Finance 재무지표]"]
    seen: set[str] = set()

    # 우선순위 지표 먼저 출력
    for pkey in _PRIORITY:
        for k, v in merged.items():
            if pkey in k and k not in seen:
                lines.append(f"  {k}: {v}")
                seen.add(k)

    # 나머지 관련 지표
    for k, v in merged.items():
        if k not in seen and any(kw in k for kw in _KEEP_KEYWORDS):
            lines.append(f"  {k}: {v}")
            seen.add(k)

    if len(lines) == 1:
        return f"{label} 재무지표를 파싱할 수 없었습니다 (Naver Finance 페이지 구조 변경 가능성)."

    return "\n".join(lines)
