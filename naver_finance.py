"""
naver_finance.py - Naver 증권 재무정보 스크래핑

get_financials(identifier) 함수 하나로 종목명/코드를 받아
PER, PBR, EPS, BPS, 시가총액, 매출, 영업이익 등을 텍스트로 반환.
"""

import logging
import re
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

# STOCKS 목록에 없는 주요 종목 이름 → 6자리 코드 정적 매핑
_WELL_KNOWN_STOCKS: dict[str, str] = {
    "삼성전자": "005930", "SK하이닉스": "000660", "하이닉스": "000660",
    "LG에너지솔루션": "373220", "삼성바이오로직스": "207940",
    "현대차": "005380", "현대자동차": "005380", "기아": "000270",
    "POSCO홀딩스": "005490", "포스코홀딩스": "005490", "포스코": "005490",
    "셀트리온": "068270", "KB금융": "105560", "신한지주": "055550",
    "삼성SDI": "006400", "LG화학": "051910",
    "카카오": "035720", "NAVER": "035420", "네이버": "035420",
    "삼성물산": "028260", "SK이노베이션": "096770",
    "SK텔레콤": "017670", "KT": "030200", "한국전력": "015760",
    "삼성생명": "032830", "삼성화재": "000810",
    "우리금융지주": "316140", "메리츠금융지주": "138040",
    "롯데케미칼": "011170", "현대건설": "000720",
    "삼성전기": "009150", "LG디스플레이": "034220",
    "SK": "034730", "LG": "003550",
    "현대글로비스": "086280", "포스코퓨처엠": "003670",
    "HD현대중공업": "329180", "현대제철": "004020",
    "한국항공우주": "047810", "한국항공우주산업": "047810",
    "대한항공": "003490", "에쓰오일": "010950",
    "한화에어로스페이스": "012450", "한화솔루션": "009830",
    "크래프톤": "259960", "카카오뱅크": "323410", "카카오페이": "377300",
    "고려아연": "010130", "포스코인터내셔널": "047050",
    "HD현대": "267250", "두산밥캣": "241560",
    "현대해상": "001450", "DB손해보험": "005830",
}

# 우선 출력할 지표 순서
_PRIORITY = [
    "현재주가",
    "시가총액", "PER", "EPS", "추정PER", "추정EPS",
    "PBR", "BPS", "주당배당금", "시가배당률", "ROE",
    "매출액", "영업이익", "당기순이익", "부채비율", "유동비율",
]

# 이 키워드가 포함된 항목만 필터링
_KEEP_KEYWORDS = [
    "현재주가", "현재가",
    "PER", "EPS", "PBR", "BPS", "ROE", "시가총액", "배당",
    "매출", "영업이익", "당기순이익", "부채", "유동비율",
    "상장주식", "외국인",
]


def _resolve_code(identifier: str) -> tuple[str, str]:
    """종목명 또는 티커 → (6자리코드, 종목명). 실패 시 ('', '')."""
    from config import STOCKS

    clean = identifier.replace(".KS", "").replace(".KQ", "").strip()

    # 1. STOCKS 딕셔너리에서 탐색
    for ticker, name in STOCKS.items():
        code = ticker.replace(".KS", "").replace(".KQ", "")
        if identifier.strip() in (ticker, code, name) or clean in (code, name):
            return code, name

    # 2. 정적 well-known 종목 매핑에서 탐색
    if clean in _WELL_KNOWN_STOCKS:
        return _WELL_KNOWN_STOCKS[clean], clean

    # 3. identifier 자체가 6자리 숫자 코드인 경우
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

    # ── 현재주가 명시적 추출 (CSS 셀렉터 우선, 미확보 시 기파싱 결과 재활용) ──
    if "현재주가" not in result:
        for selector in ["em#_nowVal", ".no_today em", "#_priceValue"]:
            el = soup.select_one(selector)
            if el:
                for span in el.find_all("span", class_="blind"):
                    span.decompose()
                price_text = el.get_text(strip=True).replace(",", "")
                if price_text.isdigit():
                    result["현재주가"] = f"{int(price_text):,}원"
                    break
    if "현재주가" not in result:
        for k in list(result.keys()):
            if k in ("현재가", "주가"):
                result["현재주가"] = result[k]
                break

    return result


def _scrape_coinfo(code: str) -> dict:
    """Naver Finance 기업정보 페이지 → 연도별 재무 요약 딕셔너리 반환."""
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

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue

        # 연도 헤더 행 탐색 (20XX 패턴이 2개 이상인 행)
        year_labels: list[str] = []
        header_idx = -1
        for i, row in enumerate(rows):
            ths = row.find_all("th")
            if not ths:
                continue
            texts = [th.get_text(strip=True) for th in ths]
            if sum(1 for t in texts if re.search(r"20\d{2}", t)) >= 2:
                year_labels = texts
                header_idx = i
                break

        if not year_labels or header_idx == -1:
            continue

        # 첫 번째 열은 항목명, 나머지가 연도 라벨
        years = year_labels[1:]

        for row in rows[header_idx + 1:]:
            th = row.find("th")
            tds = row.find_all("td")
            if not th or not tds:
                continue
            metric = th.get_text(strip=True)
            if not metric:
                continue
            for j, td in enumerate(tds):
                if j >= len(years):
                    break
                val = td.get_text(strip=True)
                if val and val not in ("-", "N/A", ""):
                    result.setdefault(f"{metric}({years[j]})", val)

    return result


def _scrape_wisereport(code: str) -> dict:
    """wisereport에서 연도별 주요지표(PER/PBR/EPS/BPS 등) 파싱."""
    url = f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("wisereport 조회 실패 (%s): %s", code, e)
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.select("table")
    result: dict[str, str] = {}

    # table[1] — 주가/시가총액 기본 정보
    if len(tables) > 1:
        for row in tables[1].select("tr"):
            cells = row.select("td,th")
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True)
                val = cells[1].get_text(strip=True)
                if key and val and val not in ("-", ""):
                    result.setdefault(key, val)

    # table[5] — 연도별 주요지표 (PER, PBR, EPS, BPS 등)
    if len(tables) > 5:
        rows = tables[5].select("tr")
        if rows:
            header_cells = rows[0].select("th,td")
            years = [c.get_text(strip=True) for c in header_cells[1:]]
            for row in rows[1:]:
                cells = row.select("th,td")
                if not cells:
                    continue
                metric = cells[0].get_text(strip=True)
                if not metric:
                    continue
                for i, year in enumerate(years):
                    if i + 1 < len(cells):
                        val = cells[i + 1].get_text(strip=True)
                        if val and val not in ("-", "N/A", ""):
                            result[f"{metric}({year})"] = val

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

    main_data   = _scrape_main(code)
    coinfo_data = _scrape_coinfo(code)
    wise_data   = _scrape_wisereport(code)
    # 우선순위: coinfo < wisereport < main (현재가/당일 데이터는 main이 최신)
    # wisereport의 연도별 라벨 키(EPS(2026/12(E)) 등)는 main에 없으므로 보존됨
    merged = {**coinfo_data, **wise_data, **main_data}

    if not merged:
        return f"{label} 재무정보 조회 실패 — Naver Finance 연결 오류"

    lines = [f"[{label} — Naver Finance 재무지표]"]
    seen: set[str] = set()

    # wisereport 연도별 지표를 최상단에 먼저 출력 (GPT가 연도별 EPS 혼동 방지)
    year_keys = [(k, v) for k, v in wise_data.items() if re.search(r"\d{4}/\d{2}\(", k)]
    if year_keys:
        lines.append("  ── 연도별 주요지표 (wisereport) ──")
        for k, v in year_keys:
            lines.append(f"  {k}: {v}")
            seen.add(k)

    # 우선순위 지표
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
