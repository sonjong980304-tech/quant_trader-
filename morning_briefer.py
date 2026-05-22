"""
morning_briefer.py - 평일 오전 8시 자동 모닝 브리핑

흐름:
  1. Tavily 웹 검색으로 미국 증시 / 관심종목 뉴스 / 경제 캘린더 수집
  2. GPT-A(gpt-4o-mini): 4가지 질문에 대한 브리핑 생성
  3. GPT-B(gpt-4o-mini): Context Recall 평가 (검색 결과를 얼마나 충실히 반영했는지)
  4. 브리핑 + 평가 결과를 텔레그램으로 전송
"""

import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytz
from openai import OpenAI
from tavily import TavilyClient

from config import STOCKS, OPENAI_API_KEY, TELEGRAM_CHAT_ID
from notifier import send_telegram

logger = logging.getLogger(__name__)
KST    = pytz.timezone("Asia/Seoul")


# ─────────────────────────────────────────────
# 웹 검색
# ─────────────────────────────────────────────

def _search(query: str, k: int = 5, days: int = 2) -> str:
    """Tavily 검색 결과를 문자열로 반환 (days: 최근 N일 이내 결과만)"""
    import os
    client  = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
    results = client.search(
        query,
        max_results=k,
        days=days,
        search_depth="advanced",
    )["results"]
    return "\n\n".join(
        f"[출처: {r['url']}]\n{r['content']}"
        for r in results
    )


def _gather_context() -> dict:
    """미국 증시 / 관심종목 뉴스 / 경제 캘린더를 병렬 검색"""
    now        = datetime.now(KST)
    # KST 8시 기준 미국 전날 장 마감 → US 날짜는 어제
    from datetime import timedelta
    us_date    = (now - timedelta(days=1)).strftime("%B %d %Y")   # e.g. "May 21 2026"
    kst_today  = now.strftime("%Y-%m-%d")
    stock_list = " OR ".join(STOCKS.values())

    queries = {
        "us_market":     f"US stock market close {us_date} S&P500 Nasdaq Composite Dow Jones recap results",
        "stock_news":    f"{kst_today} ({stock_list}) 주식 뉴스",
        "econ_calendar": f"economic calendar {now.strftime('%B %Y')} upcoming events this week CPI FOMC schedule",
    }

    context = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_search, q): key for key, q in queries.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                context[key] = future.result()
            except Exception as e:
                logger.warning("검색 실패 (%s): %s", key, e)
                context[key] = "검색 결과 없음"

    return context


# ─────────────────────────────────────────────
# GPT-A: 브리핑 생성
# ─────────────────────────────────────────────

def _generate_briefing(context: dict) -> str:
    """검색 결과를 바탕으로 4가지 질문에 답하는 브리핑 생성"""
    client     = OpenAI(api_key=OPENAI_API_KEY)
    stock_list = ", ".join(STOCKS.values())

    user_prompt = f"""아래 검색 결과를 바탕으로 다음 4가지 질문에 답해줘.

=== 미국 증시 데이터 ===
{context.get("us_market", "")}

=== 관심종목 뉴스 ===
{context.get("stock_news", "")}

=== 경제 캘린더 ===
{context.get("econ_calendar", "")}

---

1. 간밤 미국 증시 마감시황을 요약해줘. 다우존스, S&P500, 나스닥 각각의 등락률을 반드시 포함하고, 상승/하락의 원인을 한 가지만 짚어줘.
2. 위의 결과를 바탕으로, 오늘 한국 주식 시장의 개장 분위기를 긍정, 중립, 부정 중 하나로 판단하고 그 이유를 설명해줘.
3. 내 현재 관심종목({stock_list})과 직접 관련된 핵심 뉴스가 있다면 한 개씩만 요약해줘. 뉴스가 없으면 "특이사항 없음"으로 보고해.
4. 오늘을 기준으로 발표가 예정된 시장에 영향을 줄 수 있는 주요 경제 지표나 이벤트가 있다면 시간(한국 시간 기준)과 함께 알려줘.
"""

    resp = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "system", "content": "당신은 전문 금융 시황 브리핑 어시스턴트입니다. 간결하고 정확하게 답변하세요."},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


# ─────────────────────────────────────────────
# GPT-B: Context Recall 평가
# ─────────────────────────────────────────────

def _evaluate_context_recall(briefing: str, context: dict) -> tuple:
    """
    검색 컨텍스트 대비 브리핑의 Context Recall을 평가.
    - 점수(0.0~1.0): 컨텍스트 핵심 정보 중 브리핑에 반영된 비율
    - 평가 코멘트: 잘 반영된 점 / 누락된 점
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 컨텍스트가 너무 길면 잘라서 평가에 사용
    ctx_text = (
        f"[미국 증시]\n{context.get('us_market', '')[:1500]}\n\n"
        f"[경제 캘린더]\n{context.get('econ_calendar', '')[:800]}"
    )

    eval_prompt = f"""당신은 RAG 품질 평가 전문가입니다. Context Recall을 평가해주세요.

Context Recall이란: 검색된 컨텍스트의 핵심 정보 중 실제 답변에 반영된 비율입니다.

=== 검색된 컨텍스트 ===
{ctx_text}

=== 생성된 브리핑 ===
{briefing}

---
위 브리핑이 컨텍스트의 핵심 정보를 얼마나 충실히 반영했는지 평가해주세요.

반드시 아래 형식으로만 답변하세요:
점수: [0.0~1.0 사이의 숫자]
잘된 점: [한 문장]
아쉬운 점: [한 문장, 없으면 "없음"]
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0,
    )
    content = resp.choices[0].message.content.strip()

    # 점수 파싱
    score = 0.0
    for line in content.splitlines():
        if line.startswith("점수:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass

    return score, content


# ─────────────────────────────────────────────
# 시가총액 랭킹
# ─────────────────────────────────────────────

# Nasdaq 시가총액 상위 후보 (15개 중 상위 10개 선별)
_NASDAQ_CANDIDATES = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA",
    "AVGO","COST","NFLX","AMD","ADBE","QCOM","TXN","INTC",
]

# 다우존스 30 구성 종목
_DOW_COMPONENTS = [
    "AAPL","AMGN","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS",
    "DOW","GS","HD","HON","IBM","JNJ","JPM","KO","MCD","MMM",
    "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WMT","INTC",
]


def _fetch_us_rankings(tickers: list, label: str, flag: str) -> str:
    """yfinance로 미국 시장 시가총액 Top 10 조회 (병렬)"""
    import yfinance as yf

    def _info(ticker):
        try:
            t    = yf.Ticker(ticker)
            fi   = t.fast_info
            name = t.info.get("shortName", ticker)
            return {
                "ticker": ticker,
                "name":   name,
                "mc":     fi.market_cap or 0,
                "price":  fi.last_price or 0,
            }
        except Exception:
            return {"ticker": ticker, "name": ticker, "mc": 0, "price": 0}

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_info, tickers))

    results.sort(key=lambda x: x["mc"], reverse=True)
    lines = [f"<b>{flag} {label} 시가총액 Top 10</b>"]
    for rank, d in enumerate(results[:10], 1):
        mc_t = d["mc"] / 1e12
        lines.append(f"{rank}. {d['name']} ({d['ticker']}) — ${d['price']:,.1f} | ${mc_t:.2f}T")
    return "\n".join(lines)


_KOSPI_CANDIDATES = [
    "005930.KS","000660.KS","373220.KS","207940.KS","005380.KS",
    "000270.KS","051910.KS","006400.KS","035420.KS","035720.KS",
    "068270.KS","105560.KS","055550.KS","086790.KS","096770.KS",
    "017670.KS","003550.KS","015760.KS","032830.KS","030200.KS",
]

_KOSDAQ_CANDIDATES = [
    "247540.KQ","086520.KQ","196170.KQ","041510.KQ","263750.KQ",
    "122870.KQ","035900.KQ","145020.KQ","357780.KQ","294870.KQ",
    "028300.KQ","214150.KQ","046310.KQ","900140.KQ","950130.KQ",
]


def _fetch_kr_rankings(tickers: list, label: str) -> str:
    """yfinance로 한국 시장 시가총액 Top 10 조회"""
    import yfinance as yf

    def _info(ticker):
        try:
            t    = yf.Ticker(ticker)
            fi   = t.fast_info
            info = t.info
            name = info.get("shortName") or info.get("longName", ticker)
            return {
                "ticker": ticker,
                "name":   name,
                "mc":     fi.market_cap or 0,
                "price":  fi.last_price or 0,
            }
        except Exception:
            return {"ticker": ticker, "name": ticker, "mc": 0, "price": 0}

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_info, tickers))

    results.sort(key=lambda x: x["mc"], reverse=True)
    lines = [f"<b>🇰🇷 {label} 시가총액 Top 10</b>"]
    for rank, d in enumerate(results[:10], 1):
        mc_t = d["mc"] / 1e12
        lines.append(f"{rank}. {d['name']} ({d['ticker']}) — {mc_t:.1f}조원")
    return "\n".join(lines)


def fetch_market_rankings() -> str:
    """KOSPI / KOSDAQ / Nasdaq / 다우존스 시가총액 Top 10 병렬 조회"""
    tasks = {
        "kospi":   lambda: _fetch_kr_rankings(_KOSPI_CANDIDATES,  "KOSPI"),
        "kosdaq":  lambda: _fetch_kr_rankings(_KOSDAQ_CANDIDATES, "KOSDAQ"),
        "nasdaq":  lambda: _fetch_us_rankings(_NASDAQ_CANDIDATES, "Nasdaq",  "🇺🇸"),
        "dow":     lambda: _fetch_us_rankings(_DOW_COMPONENTS,    "다우존스", "🇺🇸"),
    }

    sections = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                sections[key] = future.result()
            except Exception as e:
                logger.warning("랭킹 조회 실패 (%s): %s", key, e)
                sections[key] = f"⚠️ {key} 조회 실패"

    order = ["kospi", "kosdaq", "nasdaq", "dow"]
    return "\n\n".join(sections[k] for k in order if k in sections)


# ─────────────────────────────────────────────
# 메인 진입점
# ─────────────────────────────────────────────

def send_morning_briefing():
    """평일 오전 8시 자동 실행 — 모닝 브리핑 수집·생성·평가 후 텔레그램 전송"""
    now = datetime.now(KST)
    if now.weekday() >= 5:
        return

    logger.info("모닝 브리핑 시작 (%s)", now.strftime("%Y-%m-%d %H:%M"))

    try:
        # 1. 컨텍스트 수집 (병렬 검색)
        logger.info("  웹 검색 중...")
        context = _gather_context()

        # 2. GPT-A: 브리핑 생성
        logger.info("  브리핑 생성 중...")
        briefing = _generate_briefing(context)

        # 3. GPT-B: Context Recall 평가 (브리핑 생성과 병렬 실행 가능하나 브리핑 의존)
        logger.info("  Context Recall 평가 중...")
        score, evaluation = _evaluate_context_recall(briefing, context)

        # 4. 시가총액 랭킹 조회 (브리핑 평가와 병렬)
        logger.info("  시가총액 랭킹 조회 중...")
        rankings = fetch_market_rankings()

        # 5. 텔레그램 전송
        send_telegram(
            f"🌅 <b>모닝 브리핑</b> {now.strftime('%Y-%m-%d %H:%M')}"
        )
        send_telegram(briefing)
        send_telegram(f"📊 <b>시가총액 Top 10</b>\n\n{rankings}")
        send_telegram(
            f"📋 <b>브리핑 품질 평가 (Context Recall)</b>\n{evaluation}"
        )

        logger.info("  모닝 브리핑 완료 (recall=%.2f)", score)

    except Exception as e:
        logger.error("모닝 브리핑 실패: %s", e)
        send_telegram(f"⚠️ 모닝 브리핑 오류: {e}")
