"""
langchain_agent.py - LangGraph ReAct 기반 주식 어시스턴트

AgentExecutor → create_react_agent (langgraph.prebuilt) 전환.
MemorySaver checkpointer로 thread_id(유저별) 대화 이력 분리 관리.
clear_history()는 generation 카운터를 증가시켜 새 thread로 전환.
"""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from config import OPENAI_API_KEY, STOCKS, MA_SHORT, MA_LONG

logger = logging.getLogger(__name__)

# MemorySaver: thread_id 기반으로 유저별 대화 이력 분리
_checkpointer      = MemorySaver()
_react_agent       = None
# clear_history() 시 generation 증가 → 새 thread_id로 전환
_thread_generation: dict[int, int] = {}


# ─────────────────────────────────────────────
# 도구 구현 함수 (gpt_agent.py에서 이전)
# ─────────────────────────────────────────────

def _parse_korean_date(date_str: str):
    """한국어/일반 날짜 문자열 → datetime.date. 실패 시 None."""
    import re
    from datetime import date
    s = date_str.strip()
    for pattern, fmt in [
        (r"(\d{4})-(\d{1,2})-(\d{1,2})", lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
        (r"(\d{4})\.(\d{1,2})\.(\d{1,2})", lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
        (r"(\d{2,4})년\s*(\d{1,2})월\s*(\d{1,2})일", lambda m: date(int(m[1]) if int(m[1]) > 100 else 2000 + int(m[1]), int(m[2]), int(m[3]))),
        (r"(\d{8})", lambda m: date(int(m[1][:4]), int(m[1][4:6]), int(m[1][6:8]))),
    ]:
        match = re.search(pattern, s)
        if match:
            try:
                return fmt(match)
            except ValueError:
                continue
    return None


def _call_yahoo_finance(ticker: str) -> str:
    try:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())
        info = t.info

        if not info or info.get("quoteType") is None:
            return f"'{ticker}' 종목을 Yahoo Finance에서 찾을 수 없습니다."

        name     = info.get("longName") or info.get("shortName") or ticker
        currency = info.get("currency", "USD")

        def fmt_num(v, unit="", pct=False):
            if v is None:
                return "N/A"
            if pct:
                return f"{v * 100:.2f}%"
            if abs(v) >= 1e12:
                return f"{v / 1e12:.2f}T {currency}{unit}"
            if abs(v) >= 1e9:
                return f"{v / 1e9:.2f}B {currency}{unit}"
            if abs(v) >= 1e6:
                return f"{v / 1e6:.2f}M {currency}{unit}"
            return f"{v:,.2f}{unit}"

        lines = [f"[{name} ({ticker.upper()}) — Yahoo Finance 재무지표]", ""]
        lines.append("● 주가")
        lines.append(f"  현재가      : {fmt_num(info.get('currentPrice'))} {currency}")
        lines.append(f"  52주 고/저  : {fmt_num(info.get('fiftyTwoWeekHigh'))} / {fmt_num(info.get('fiftyTwoWeekLow'))}")
        lines.append(f"  시가총액    : {fmt_num(info.get('marketCap'))}")
        lines.append("\n● 밸류에이션")
        lines.append(f"  PER (TTM)   : {fmt_num(info.get('trailingPE'))}")
        lines.append(f"  PER (FWD)   : {fmt_num(info.get('forwardPE'))}")
        lines.append(f"  PBR         : {fmt_num(info.get('priceToBook'))}")
        lines.append(f"  EPS (TTM)   : {fmt_num(info.get('trailingEps'))} {currency}")
        lines.append(f"  EPS (FWD)   : {fmt_num(info.get('forwardEps'))} {currency}")
        lines.append("\n● 수익성")
        lines.append(f"  매출액      : {fmt_num(info.get('totalRevenue'))}")
        lines.append(f"  영업이익률  : {fmt_num(info.get('operatingMargins'), pct=True)}")
        lines.append(f"  순이익률    : {fmt_num(info.get('profitMargins'), pct=True)}")
        lines.append(f"  ROE         : {fmt_num(info.get('returnOnEquity'), pct=True)}")
        lines.append(f"  ROA         : {fmt_num(info.get('returnOnAssets'), pct=True)}")
        lines.append("\n● 재무 건전성")
        lines.append(f"  부채비율(D/E): {fmt_num(info.get('debtToEquity'))}")
        lines.append(f"  유동비율    : {fmt_num(info.get('currentRatio'))}")
        lines.append(f"  잉여현금흐름: {fmt_num(info.get('freeCashflow'))}")
        div_yield = info.get("dividendYield")
        lines.append("\n● 배당")
        lines.append(f"  시가배당률  : {fmt_num(div_yield, pct=True)}")
        lines.append(f"  주당배당금  : {fmt_num(info.get('dividendRate'))} {currency}")
        try:
            fin = t.financials
            if fin is not None and not fin.empty:
                lines.append("\n● 연간 실적 (최근 3년)")
                cols = fin.columns[:3]
                for row_key, label in [
                    ("Total Revenue",    "매출액"),
                    ("Operating Income", "영업이익"),
                    ("Net Income",       "순이익"),
                ]:
                    if row_key in fin.index:
                        vals = "  |  ".join(
                            f"{str(c)[:4]}: {fmt_num(fin.loc[row_key, c])}"
                            for c in cols
                        )
                        lines.append(f"  {label}: {vals}")
        except Exception:
            pass
        return "\n".join(lines)
    except Exception as e:
        logger.error("Yahoo Finance 툴 오류: %s", e)
        return f"Yahoo Finance 조회 오류: {e}"


def _call_naver_news(query: str, n: int = 5) -> str:
    try:
        from news_fetcher import fetch_naver_news
        n = min(max(1, n), 10)
        items = fetch_naver_news(query, n=n)
        if not items:
            return f"'{query}' 관련 뉴스를 찾을 수 없습니다. (NAVER API 키 확인 필요)"
        lines = [f"[{query} — 네이버 최신 뉴스 {len(items)}건]"]
        for i, item in enumerate(items, 1):
            title = item.get("title", "제목 없음")
            link  = item.get("link", "")
            pub   = item.get("pubDate", "")
            desc  = item.get("description", "")
            lines.append(f"\n{i}. {title}")
            if pub:
                lines.append(f"   📅 {pub}")
            if desc:
                lines.append(f"   {desc[:100]}{'...' if len(desc) > 100 else ''}")
            if link:
                lines.append(f"   🔗 {link}")
        return "\n".join(lines)
    except Exception as e:
        logger.error("네이버 뉴스 툴 오류: %s", e)
        return f"네이버 뉴스 조회 오류: {e}"


def _call_historical_price(identifier: str, date: str) -> str:
    try:
        import yfinance as yf
        import pandas as pd
        from datetime import timedelta
        from naver_finance import _resolve_code

        code, name = _resolve_code(identifier)
        if not code:
            return f"'{identifier}' 종목을 찾을 수 없습니다."

        target = _parse_korean_date(date)
        if not target:
            return f"날짜 형식을 인식할 수 없습니다: {date}\n'YYYY-MM-DD' 또는 'YY년 M월 D일' 형식으로 입력하세요."

        label = name or code
        start = (target - timedelta(days=7)).strftime("%Y-%m-%d")
        end   = (target + timedelta(days=7)).strftime("%Y-%m-%d")

        ticker = f"{code}.KS"
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            ticker = f"{code}.KQ"
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            return f"{label} {date} 주가 데이터를 가져올 수 없습니다."

        df.index = pd.to_datetime(df.index).normalize()
        target_ts = pd.Timestamp(target)

        if target_ts in df.index:
            close = float(df.loc[target_ts, "Close"].iloc[0] if hasattr(df.loc[target_ts, "Close"], "iloc") else df.loc[target_ts, "Close"])
            return f"{label} {target.strftime('%Y-%m-%d')} 종가: {close:,.0f}원"

        past = df[df.index <= target_ts]
        row_ts = past.index[-1] if not past.empty else df.index[0]
        close = float(df.loc[row_ts, "Close"].iloc[0] if hasattr(df.loc[row_ts, "Close"], "iloc") else df.loc[row_ts, "Close"])
        return (
            f"{label} {row_ts.strftime('%Y-%m-%d')} 종가: {close:,.0f}원"
            f" (요청일 {target.strftime('%Y-%m-%d')} 기준 가장 가까운 거래일)"
        )
    except Exception as e:
        logger.error("과거 주가 조회 오류: %s", e)
        return f"과거 주가 조회 오류: {e}"


def _call_stock_signal(identifier: str) -> str:
    try:
        from config import (
            STOCKS, MA_SHORT, MA_LONG, RSI_PERIOD,
            VOLUME_LOOKBACK_DAYS, VOLUME_INCREASE_RATIO, VOLUME_SURGE_RATIO,
        )
        from data_fetcher import fetch_ohlcv
        from indicators import add_all_indicators, detect_crossover
        from strategy import generate_signals, get_latest_signal

        ticker = None
        stock_name = identifier
        clean = identifier.replace(".KS", "").replace(".KQ", "").strip()

        for t, name in STOCKS.items():
            code = t.replace(".KS", "").replace(".KQ", "")
            if identifier.strip() in (t, code, name) or clean in (code, name):
                ticker = t
                stock_name = name
                break

        if not ticker:
            if clean.isdigit() and len(clean) == 6:
                import yfinance as yf
                for suffix in [".KS", ".KQ"]:
                    df_test = yf.download(f"{clean}{suffix}", period="5d", progress=False, auto_adjust=True)
                    if not df_test.empty:
                        ticker = f"{clean}{suffix}"
                        stock_name = clean
                        break
            if not ticker:
                return f"'{identifier}' 종목을 찾을 수 없습니다."

        stock_code = ticker.replace(".KS", "").replace(".KQ", "")
        df = fetch_ohlcv(ticker, period_years=1)
        try:
            from runner import _append_today_bar
            from data_fetcher import get_minute_data
            minute_df = None
            try:
                minute_df = get_minute_data(ticker, interval_min=1)
            except Exception:
                pass
            df = _append_today_bar(df, minute_df)
        except Exception:
            pass

        df = add_all_indicators(df, short=MA_SHORT, long=MA_LONG, rsi_period=RSI_PERIOD)
        df = detect_crossover(df, short=MA_SHORT, long=MA_LONG)
        df = generate_signals(df)
        sig = get_latest_signal(df)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        close      = sig["close"]
        ma5        = sig["ma_short"]
        ma20       = sig["ma_long"]
        rsi        = sig["rsi"]
        volume     = sig["volume"]
        open_price = float(last["Open"])
        prev_close = float(prev["Close"])
        prev_ma5   = float(prev[f"MA_{MA_SHORT}"])
        prev_ma20  = float(prev[f"MA_{MA_LONG}"])
        ma_min     = min(prev_ma5, prev_ma20)
        ma_max     = max(prev_ma5, prev_ma20)
        vol_avg    = float(df["Volume"].rolling(window=VOLUME_LOOKBACK_DAYS, min_periods=1).mean().shift(1).iloc[-1])
        vol_ratio  = volume / vol_avg if vol_avg > 0 else 0
        candle     = "양봉" if close > open_price else ("음봉" if close < open_price else "도지")

        def chk(cond): return "✅" if cond else "❌"

        c1_a = prev_close > prev_ma5
        c1_b = float(last["Low"]) < open_price
        c1_c = close > open_price
        c1   = bool(last.get("buy_signal_1", False))
        c2_a = ma_min < prev_close < ma_max
        c2_b = vol_ratio >= VOLUME_INCREASE_RATIO
        c2_c = close > open_price
        c2   = bool(last.get("buy_signal_2", False))
        c3_a = prev_close < prev_ma20
        c3_b = vol_ratio >= VOLUME_SURGE_RATIO
        c3_c = close >= open_price
        c3   = bool(last.get("buy_signal_3", False))

        lines = [
            f"[{stock_name} ({stock_code}) — {sig['date']} 기준 신호 분석]", "",
            f"● 현재가: {close:,.0f}원  MA5: {ma5:,.0f}  MA20: {ma20:,.0f}  RSI: {rsi}",
            f"● 캔들: {candle} (시가 {open_price:,.0f}원)",
            f"● 거래량: {volume:,}주  ({vol_ratio:.2f}배 / {VOLUME_LOOKBACK_DAYS}일 평균 {vol_avg:,.0f}주)", "",
            "▶ 1원칙 (시가돌파)",
            f"  {chk(c1_a)} 전일 종가({prev_close:,.0f}) > MA5({prev_ma5:,.0f})",
            f"  {chk(c1_b)} 당일 저가({float(last['Low']):,.0f}) < 시가({open_price:,.0f})",
            f"  {chk(c1_c)} 당일 종가 > 시가 (양봉)",
            f"  → {'✅ 신호 발생' if c1 else '❌ 미발생'}", "",
            "▶ 2원칙 (MA사이반등)",
            f"  {chk(c2_a)} 전일 종가({prev_close:,.0f})가 MA5({prev_ma5:,.0f})~MA20({prev_ma20:,.0f}) 사이",
            f"  {chk(c2_b)} 거래량 비율 {vol_ratio:.2f}배 ≥ {VOLUME_INCREASE_RATIO}배 기준",
            f"  {chk(c2_c)} 양봉",
            f"  → {'✅ 신호 발생' if c2 else '❌ 미발생'}", "",
            "▶ 3원칙 (MA20아래급등)",
            f"  {chk(c3_a)} 전일 종가({prev_close:,.0f}) < MA20({prev_ma20:,.0f})",
            f"  {chk(c3_b)} 거래량 비율 {vol_ratio:.2f}배 ≥ {VOLUME_SURGE_RATIO}배 기준",
            f"  {chk(c3_c)} 양봉 또는 도지",
            f"  → {'✅ 신호 발생' if c3 else '❌ 미발생'}", "",
            "━━━ 종합 신호 ━━━",
        ]
        if sig["buy"]:
            lines.append(f"✅ 매수: {'/'.join(sig['buy_which'])}")
        elif sig["sell_partial"]:
            lines.append(f"🟡 매도(부분): {'/'.join(sig['sell_which'])}")
        elif sig["sell_full"]:
            lines.append(f"🔴 매도(전량): {'/'.join(sig['sell_which'])}")
        else:
            lines.append("⚪ 매수/매도 신호 없음")
        return "\n".join(lines)
    except Exception as e:
        logger.error("종목 신호 분석 오류: %s", e)
        return f"종목 신호 분석 오류: {e}"


def _call_set_conditional_order(stock_name: str, condition_type: str,
                                condition_value: float, action: str, quantity: int,
                                stock_code: str = "") -> str:
    try:
        from conditional_orders import add_order
        ticker = ""

        if stock_code:
            stock_code = stock_code.strip().replace(".KS", "").replace(".KQ", "")
            if not stock_name:
                try:
                    from pykrx import stock as krx
                    stock_name = krx.get_market_ticker_name(stock_code) or stock_name
                except Exception:
                    pass
            suffix = ".KQ" if stock_code.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")) else ".KS"
            for t in STOCKS:
                if t.replace(".KS", "").replace(".KQ", "") == stock_code:
                    suffix = ".KS" if t.endswith(".KS") else ".KQ"
                    break
            else:
                try:
                    from pykrx import stock as krx
                    from datetime import date, timedelta
                    check_date = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
                    kosdaq = krx.get_market_ticker_list(check_date, market="KOSDAQ")
                    suffix = ".KQ" if stock_code in kosdaq else ".KS"
                except Exception:
                    suffix = ".KS"
            ticker = stock_code + suffix

        if not ticker:
            for t, name in STOCKS.items():
                if name == stock_name:
                    ticker = t
                    stock_code = t.replace(".KS", "").replace(".KQ", "")
                    break

        if not ticker:
            try:
                from pykrx import stock as krx
                from datetime import date, timedelta
                check_date = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
                for market in ("KOSPI", "KOSDAQ"):
                    tickers = krx.get_market_ticker_list(check_date, market=market)
                    for code in tickers:
                        nm = krx.get_market_ticker_name(code)
                        if nm == stock_name:
                            suffix = ".KS" if market == "KOSPI" else ".KQ"
                            ticker = code + suffix
                            stock_code = code
                            break
                    if ticker:
                        break
            except Exception:
                pass

        if not ticker:
            return (
                f"'{stock_name}' 종목을 찾을 수 없습니다.\n"
                f"종목코드(6자리)를 함께 알려주시면 바로 등록됩니다. (예: '삼성전자 005930 30만원 이하 1주 매수')"
            )

        order = add_order(ticker, stock_name, stock_code,
                          condition_type, condition_value, action, quantity)
        ctype_label = {
            "price_below":  f"현재가 < {condition_value:,.0f}원",
            "price_above":  f"현재가 > {condition_value:,.0f}원",
            "profit_above": f"수익률 > +{condition_value}%",
            "profit_below": f"수익률 < {condition_value}%",
        }.get(condition_type, condition_type)
        action_label = {"buy": "매수", "sell": "매도", "sellall": "전량 매도"}.get(action, action)
        qty_str = f"{quantity}주 " if action != "sellall" else ""
        return (
            f"✅ 조건부 주문 등록 완료 [#{order['id']}]\n"
            f"  종목: {stock_name}({stock_code})\n"
            f"  조건: {ctype_label}\n"
            f"  실행: {action_label} {qty_str}(시장가)\n"
            f"  5분 간격으로 조건 체크 후 자동 실행됩니다."
        )
    except Exception as e:
        logger.error("조건부 주문 등록 오류: %s", e)
        return f"조건부 주문 등록 실패: {e}"


def _call_list_conditional_orders() -> str:
    try:
        from conditional_orders import list_orders
        orders = list_orders()
        if not orders:
            return "등록된 조건부 주문이 없습니다."
        lines = [f"[조건부 주문 목록 — {len(orders)}건]"]
        for o in orders:
            ctype_label = {
                "price_below":  f"현재가 < {float(o['condition_value']):,.0f}원",
                "price_above":  f"현재가 > {float(o['condition_value']):,.0f}원",
                "profit_above": f"수익률 > +{o['condition_value']}%",
                "profit_below": f"수익률 < {o['condition_value']}%",
            }.get(o["condition_type"], o["condition_type"])
            action_label = {"buy": "매수", "sell": "매도", "sellall": "전량 매도"}.get(o["action"], o["action"])
            qty_str = f"{o['quantity']}주 " if o["action"] != "sellall" else ""
            lines.append(
                f"  #{o['id']} | {o['stock_name']}({o['stock_code']}) | "
                f"조건: {ctype_label} → {action_label} {qty_str}| 등록: {o['created_at']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"조건부 주문 조회 실패: {e}"


def _call_cancel_conditional_order(order_id: str) -> str:
    try:
        from conditional_orders import cancel_order, list_orders
        if order_id.lower() == "all":
            orders = list_orders()
            for o in orders:
                cancel_order(o["id"])
            return f"✅ 조건부 주문 {len(orders)}건 전체 취소 완료"
        if cancel_order(order_id):
            return f"✅ 조건부 주문 #{order_id} 취소 완료"
        return f"⚠️ #{order_id}를 찾을 수 없습니다. list_conditional_orders로 ID를 확인하세요."
    except Exception as e:
        return f"조건부 주문 취소 실패: {e}"


# ─────────────────────────────────────────────
# LangChain Tools
# ─────────────────────────────────────────────

@tool
def get_naver_finance(identifier: str) -> str:
    """
    Naver 증권에서 한국 주식 재무정보(PER, PBR, ROE, EPS, 현재가 등)를 조회합니다.
    identifier: 종목명(예: '삼성전자') 또는 6자리 코드(예: '005930')
    """
    from naver_finance import get_financials
    return get_financials(identifier)


@tool
def get_yahoo_finance(ticker: str) -> str:
    """
    Yahoo Finance에서 미국 주식 재무지표(PER, EPS, ROE, 매출 등)를 조회합니다.
    ticker: Yahoo Finance 심볼 (예: 'AAPL', 'TSLA', 'NVDA')
    """
    return _call_yahoo_finance(ticker)


@tool
def get_naver_news(query: str, n: int = 5) -> str:
    """
    네이버 뉴스에서 종목 또는 키워드 관련 최신 뉴스를 검색합니다.
    query: 검색 키워드 (예: 'LG전자', '반도체 업황')
    n: 뉴스 건수 (기본 5, 최대 10)
    """
    return _call_naver_news(query, min(max(1, n), 10))


@tool
def get_stock_signal(identifier: str) -> str:
    """
    특정 종목의 현재 기술적 지표(MA, RSI, 거래량)와 매수/매도 신호를 분석합니다.
    identifier: 종목명(예: 'LG전자') 또는 6자리 코드(예: '066570')
    """
    return _call_stock_signal(identifier)


@tool
def get_historical_price(identifier: str, date: str) -> str:
    """
    특정 날짜의 종목 종가를 조회합니다.
    identifier: 종목명 또는 6자리 코드
    date: 날짜 (예: '2025-01-15', '2025년 1월 15일')
    """
    return _call_historical_price(identifier, date)


@tool
def get_account_balance() -> str:
    """현재 계좌 잔고와 보유 종목 현황을 조회합니다 (국내 + 미국주식)."""
    try:
        from trader import KISTrader
        from config import KIS_APP_KEY, IS_MOCK
        if not KIS_APP_KEY:
            return "KIS_APP_KEY 미설정 — 잔고 조회 불가"
        t       = KISTrader()
        balance = t.get_balance()
        cash    = t.get_available_cash()
        total   = t.get_total_eval_amt()
        mode    = "모의투자" if IS_MOCK else "실전투자"
        lines   = [f"[계좌 잔고 — {mode}]",
                   f"주문 가능 현금: {cash:,}원",
                   f"총평가금액(매도대금 포함): {total:,}원"]

        if balance:
            lines.append("\n[국내주식]")
            for h in balance:
                lines.append(
                    f"  • {h['name']} ({h['stock_code']}): {h['qty']}주 | "
                    f"평균 {h['avg_price']:,}원 | 평가손익 {h['eval_profit']:+,}원"
                )
        else:
            lines.append("  (국내 보유 종목 없음)")

        try:
            us_balance = t.get_us_balance()
            if us_balance:
                lines.append("\n[미국주식 — 통합증거금]")
                for h in us_balance:
                    lines.append(
                        f"  • {h['name']} ({h['symbol']}): {h['qty']}주 | "
                        f"평균 ${h['avg_price']:.2f} | 평가손익 ${h['eval_profit']:+.2f}"
                    )
            else:
                lines.append("  (미국주식 보유 없음)")
        except Exception as e:
            lines.append(f"  (미국주식 조회 실패: {e})")

        return "\n".join(lines)
    except Exception as e:
        return f"잔고 조회 실패: {e}"


@tool
def get_portfolio_status() -> str:
    """페이퍼 트레이딩 포지션 현황 (reversion + trend 슬롯 분리 10+10)을 확인합니다."""
    try:
        from paper_trader import _load, POS_PATH
        from config import REV_SLOTS, TR_SLOTS
        positions = _load(POS_PATH, {})
        rev = [p for p in positions.values() if p.get("agent") == "reversion"]
        tr  = [p for p in positions.values() if p.get("agent") == "trend"]
        lines = [
            f"[Reversion] {len(rev)}/{REV_SLOTS} 슬롯",
            *[f"  {p.get('name','?')} 진입 {p.get('entry_price',0):,.0f}원  {p.get('trade_days',0)}일" for p in rev],
            f"[Trend] {len(tr)}/{TR_SLOTS} 슬롯",
            *[f"  {p.get('name','?')} 진입 {p.get('entry_price',0):,.0f}원  {p.get('trade_days',0)}일" for p in tr],
        ]
        return "\n".join(lines) if positions else "보유 포지션 없음"
    except Exception as e:
        return f"포트폴리오 조회 실패: {e}"


@tool
def set_conditional_order(
    stock_name: str,
    condition_type: str,
    condition_value: float,
    action: str,
    quantity: int,
    stock_code: str = "",
) -> str:
    """
    조건부 주문을 등록합니다.
    condition_type: 'price_below' | 'price_above' | 'profit_above' | 'profit_below'
    action: 'buy' | 'sell' | 'sellall'
    stock_code: 6자리 종목코드 (알면 반드시 입력 — 삼성전자=005930, SK하이닉스=000660, NAVER=035420)
    """
    return _call_set_conditional_order(stock_name, condition_type, condition_value, action, quantity, stock_code)


@tool
def list_conditional_orders() -> str:
    """등록된 조건부 주문 목록을 조회합니다."""
    return _call_list_conditional_orders()


@tool
def cancel_conditional_order(order_id: str) -> str:
    """조건부 주문을 취소합니다. order_id='all'이면 전체 취소."""
    return _call_cancel_conditional_order(order_id)


@tool
def get_paper_status(market: str = "KR") -> str:
    """
    페이퍼 트레이딩 현황을 조회합니다.
    market: 'KR'(국내, 기본값) | 'US'(미국) | 'all'(전체)
    - 현재 오픈 포지션 목록과 미실현 손익
    - 현재 세션 누적 통계 (청산 건수, 승률, 기대수익률)
    """
    from paper_trader import get_metrics, _load, POS_PATH, TRADES_PATH
    import json

    mkt = None if market == "all" else market.upper()
    m   = get_metrics(mkt)

    positions = _load(POS_PATH, {})
    trades    = _load(TRADES_PATH, [])

    # 오픈 포지션 필터
    def _is_kr(t: str) -> bool:
        return t.endswith(".KS") or t.endswith(".KQ")
    if mkt == "KR":
        open_pos = {sid: p for sid, p in positions.items() if _is_kr(p.get("ticker", ""))}
    elif mkt == "US":
        open_pos = {sid: p for sid, p in positions.items() if not _is_kr(p.get("ticker", ""))}
    else:
        open_pos = positions

    lines = [f"📋 페이퍼 트레이딩 현황 ({market.upper()})"]

    # 오픈 포지션
    if open_pos:
        lines.append(f"\n【오픈 포지션 {len(open_pos)}건】")
        for p in open_pos.values():
            ep   = p.get("entry_price")
            cur  = p.get("highest", 0)
            td   = p.get("trade_days", 0)
            name = p.get("name", p.get("ticker", ""))
            ep_str = f"{ep:,.0f}원" if ep else "미확정(익일시초가)"
            lines.append(f"  • {name}: 진입가={ep_str} | 보유{td}일")
    else:
        lines.append("\n【오픈 포지션 없음】")

    # 누적 성과 (현재 세션만)
    lines.append(f"\n【누적 성과 (세션 시작: {m.get('start_date', '-')})】")
    lines.append(f"  청산 {m['n']}건 | 승률 {m['win_rate']*100:.1f}% | 기대수익 {m['ev']*100:+.2f}%")
    if m['n'] > 0:
        lines.append(f"  CI 95%: [{m['ci_low']*100:+.2f}%, {m['ci_high']*100:+.2f}%]")
        lines.append(f"  최대연속손실: {m['max_consec_loss']}연패")

    return "\n".join(lines)


@tool
def list_trade_records(status: str = "all") -> str:
    """
    매매 이력 CSV를 조회합니다.
    status: 'open'(미청산), 'closed'(청산완료), 'all'(전체)
    """
    from trade_logger import _read_all
    rows = _read_all()
    if status == "open":
        rows = [r for r in rows if not r.get("exit_date")]
    elif status == "closed":
        rows = [r for r in rows if r.get("exit_date")]
    if not rows:
        return f"({status}) 매매 이력 없음"
    lines = [f"총 {len(rows)}건:"]
    for r in rows[-20:]:  # 최근 20건
        pnl = f" | 손익 {r['pnl_pct']}%" if r.get("pnl_pct") else ""
        lines.append(
            f"[{r['trade_id']}] {r['entry_date']} {r['ticker']} {r['name']} "
            f"{r['qty']}주 진입가={r['entry_price']}{pnl}"
        )
    return "\n".join(lines)


@tool
def edit_trade_record(trade_id: str, field: str, value: str) -> str:
    """
    매매 이력 CSV의 특정 항목을 수정합니다.
    trade_id: 수정할 거래 ID(8자리) 또는 종목코드(예: 005930.KS) 또는 종목명(예: 삼성전자).
              종목코드/종목명을 주면 가장 최근 거래를 자동 선택합니다.
    field: 수정할 컬럼명 (entry_price, exit_price, qty, entry_date, exit_date, strategy, notes, win_prob, name 등)
    value: 새로운 값
    수정 불가 컬럼: trade_id, ticker, side
    """
    from trade_logger import _read_all, _write_all
    IMMUTABLE = {"trade_id", "ticker", "side"}
    if field in IMMUTABLE:
        return f"⚠️ '{field}'는 수정 불가 컬럼입니다."
    rows = _read_all()
    # 8자리 ID 직접 매칭
    target = next((r for r in rows if r["trade_id"] == trade_id), None)
    # ID 미발견 시 → 종목코드 또는 종목명으로 검색 (가장 최근 거래)
    if not target:
        key = trade_id.strip()
        candidates = [
            r for r in rows
            if r.get("ticker", "").upper() == key.upper()
            or r.get("name", "") == key
        ]
        if not candidates:
            sample = "\n".join(
                f"  [{r['trade_id']}] {r['entry_date']} {r['ticker']} {r['name']}"
                for r in rows[-5:]
            )
            return f"⚠️ '{trade_id}'를 찾을 수 없습니다. 최근 거래:\n{sample}"
        target = sorted(candidates, key=lambda r: r.get("entry_date", ""))[-1]
    old_val = target.get(field, "")
    target[field] = value
    # pnl 재계산 (entry_price 또는 exit_price 변경 시)
    if field in ("entry_price", "exit_price") and target.get("exit_price") and target.get("entry_price"):
        try:
            ep = float(target["entry_price"])
            xp = float(target["exit_price"])
            qty = int(target.get("qty", 0) or 0)
            target["pnl_pct"]    = round((xp - ep) / ep * 100, 2)
            target["pnl_amount"] = round((xp - ep) * qty, 0)
            target["win"]        = 1 if target["pnl_pct"] > 0 else 0
        except Exception:
            pass
    _write_all(rows)
    return (
        f"✅ [{trade_id}] {target['ticker']} {target['name']}\n"
        f"  {field}: '{old_val}' → '{value}'"
        + (f"\n  손익 재계산: {target.get('pnl_pct')}%" if field in ("entry_price", "exit_price") else "")
    )


# ─────────────────────────────────────────────
# 시스템 프롬프트
# ─────────────────────────────────────────────

def _build_system_prompt() -> str:
    try:
        from trader import KISTrader
        from config import KIS_APP_KEY
        if KIS_APP_KEY:
            t        = KISTrader()
            holdings = t.get_balance()
            cash     = t.get_available_cash()
            total    = t.get_total_eval_amt()
            pos_text = "\n".join(
                f"  - {h['name']} ({h['stock_code']}): {h['qty']}주 @ {h['avg_price']:,}원"
                for h in holdings
            ) or "  (보유 없음)"
            cash_text = f"  주문 가능 현금: {cash:,}원 / 총평가금액(매도대금 포함): {total:,}원"
        else:
            pos_text  = "  (KIS 미연결)"
            cash_text = "  (KIS 미연결)"
    except Exception:
        pos_text  = "  (조회 실패)"
        cash_text = "  (조회 실패)"

    stocks_text = "\n".join(f"  - {n} ({t})" for t, n in STOCKS.items()) or "  (없음)"

    return f"""당신은 퀀트 자동매매 시스템의 AI 어시스턴트입니다.
사용자의 매매 전략과 포트폴리오를 정확히 알고 있으며, 주식 관련 질문에 도움을 줍니다.
답변은 반드시 한국어로 하세요.

━━━ 툴 사용 원칙 (반드시 준수) ━━━
  현재 주가 / 재무지표(PER·PBR·ROE·EPS 등) 한국 주식 → get_naver_finance 호출
  현재 주가 / 재무지표 미국 주식 → get_yahoo_finance 호출
  특정 날짜의 과거 주가 → get_historical_price 호출
  기술적 분석·매수매도 신호 → get_stock_signal 호출
  뉴스 검색 → get_naver_news 호출
  계좌 잔고·보유 종목 → get_account_balance 호출
  페이퍼 트레이딩 현황·오픈 포지션·누적 성과 → get_paper_status 호출 (market='KR'|'US'|'all')
  매매 이력 조회 → list_trade_records 호출
  매매 이력 수정 (진입가/청산가/수량/메모 등 변경 요청 시) → edit_trade_record 호출
    ※ trade_id를 모를 경우 종목명(예: "삼성전자")이나 종목코드(예: "005930.KS")를 trade_id에 전달하면 자동 검색
  ※ 학습 데이터(training knowledge)로 주가·재무 수치를 절대 답변하지 마세요.
     반드시 툴을 호출해 실시간 데이터를 가져오세요.

━━━ 포트폴리오 구조 ━━━
  안전자산 70%: QQQ 22.3% / 삼성전자 27.3% / TLT 0.2% / ACE KRX금현물 50.3%
  급등주 30%: XGBoost ML 모델 + 켈리 공식 포지션 사이징

━━━ 매매 전략 원칙 ━━━
  MA{MA_SHORT}/MA{MA_LONG} + 거래량 + 캔들 기반 (기존 전략 유지)
  급등주: 승률 ≥ 55% AND 손익비 ≥ 1.5 조건 충족 시 알림

━━━ 계좌 현황 ━━━
{cash_text}

━━━ 현재 보유 종목 ━━━
{pos_text}

━━━ 관심 종목 ━━━
{stocks_text}"""


# ─────────────────────────────────────────────
# LangGraph ReAct Agent
# ─────────────────────────────────────────────

_TOOLS = [
    get_naver_finance,
    get_yahoo_finance,
    get_naver_news,
    get_stock_signal,
    get_paper_status,
    list_trade_records,
    edit_trade_record,
    get_historical_price,
    get_account_balance,
    get_portfolio_status,
    set_conditional_order,
    list_conditional_orders,
    cancel_conditional_order,
]


def _thread_id(user_id: int) -> str:
    """유저 ID + generation으로 thread_id 생성. clear_history() 시 generation 증가."""
    gen = _thread_generation.get(user_id, 0)
    return f"{user_id}_{gen}"


def _get_react_agent():
    """create_react_agent 싱글턴 반환 (MemorySaver 공유)."""
    global _react_agent
    if _react_agent is None:
        llm = ChatOpenAI(
            model="gpt-5.5",
            api_key=OPENAI_API_KEY,
            temperature=0,
        )
        _react_agent = create_react_agent(
            model=llm,
            tools=_TOOLS,
            prompt=_build_system_prompt(),
            checkpointer=_checkpointer,
        )
    return _react_agent


def get_agent(user_id: int):
    """유저별 agent 반환 (단일 인스턴스, thread_id로 대화 이력 분리)."""
    return _get_react_agent()


# ─────────────────────────────────────────────
# 공개 인터페이스
# ─────────────────────────────────────────────

def ask(user_id: int, question: str) -> str:
    """
    LangGraph ReAct 에이전트로 질문 처리.
    thread_id = f"{user_id}_{generation}" 으로 유저별 대화 이력 분리.
    """
    try:
        agent  = _get_react_agent()
        config = {"configurable": {"thread_id": _thread_id(user_id)}}
        result = agent.invoke(
            {"messages": [("human", question)]},
            config=config,
        )
        messages = result.get("messages", [])
        answer   = messages[-1].content if messages else ""

        # Context Recall 평가 (툴 호출이 있었을 때만)
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_msgs:
            ctx   = "\n\n".join(m.content for m in tool_msgs)
            score = _eval_recall(ctx, answer)
            return f"{answer}\n\n─────────────────\n📊 Context Recall: {score:.2f}"

        return answer
    except Exception as e:
        logger.error("LangGraph 에이전트 오류: %s", e)
        return f"⚠️ 오류: {e}"


def clear_history(user_id: int):
    """대화 기록 초기화 — generation을 증가시켜 새 thread로 전환."""
    _thread_generation[user_id] = _thread_generation.get(user_id, 0) + 1
    logger.info("유저 %d 대화 이력 초기화 (thread_id: %s)", user_id, _thread_id(user_id))


def _eval_recall(context: str, answer: str) -> float:
    """Context Recall 점수 평가 (0~1)."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        f"Context Recall = 컨텍스트로 근거를 찾을 수 있는 답변 내 주장 수 / 답변 내 전체 주장 수\n\n"
        f"[컨텍스트]\n{context[:2000]}\n\n[답변]\n{answer}\n\n"
        f"0.00~1.00 사이 숫자만 반환:"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=10,
            temperature=0,
        )
        return round(min(max(float(resp.choices[0].message.content.strip()), 0.0), 1.0), 2)
    except Exception:
        return -1.0
