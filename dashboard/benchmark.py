"""
benchmark.py - 코스피/코스닥 시가총액 가중 시장 벤치마크

전략 누적수익률과 비교할 "시장 평균" 기준선을 계산한다.
  - 가중치 : FinanceDataReader 최신 상장 시가총액(코스피/코스닥) 비중, 1일 캐시.
             일별로 변하는 값이 아니라 조회 시점 스냅샷 고정 가중치다.
  - 지수 가격 : yfinance ^KS11(코스피)/^KQ11(코스닥) 일별 종가.
"""

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=60 * 60 * 24)
def get_market_cap_weights():
    """코스피/코스닥 상장 시가총액 비중(합계 1.0). 실패 시 None."""
    try:
        import FinanceDataReader as fdr
        kospi_cap = float(fdr.StockListing("KOSPI")["Marcap"].sum())
        kosdaq_cap = float(fdr.StockListing("KOSDAQ")["Marcap"].sum())
        total = kospi_cap + kosdaq_cap
        if total <= 0:
            return None
        return {"KOSPI": kospi_cap / total, "KOSDAQ": kosdaq_cap / total}
    except Exception:
        return None


@st.cache_data(ttl=60 * 60)
def get_benchmark_curve(start: str, end: str) -> pd.DataFrame:
    """
    [start, end) 구간(YYYY-MM-DD 문자열, end는 미포함이므로 마지막 날 포함하려면
    호출측에서 +1일 해서 넘길 것)의 코스피+코스닥 시가총액가중 누적수익률(%) 곡선.
    실패·데이터 없음 시 빈 DataFrame.
    """
    weights = get_market_cap_weights()
    if not weights:
        return pd.DataFrame()
    try:
        raw = yf.download(
            ["^KS11", "^KQ11"], start=start, end=end,
            interval="1d", progress=False, auto_adjust=True,
        )
        close = raw["Close"]
    except Exception:
        return pd.DataFrame()
    if close.empty or "^KS11" not in close.columns or "^KQ11" not in close.columns:
        return pd.DataFrame()
    close = close.dropna()
    if close.empty:
        return pd.DataFrame()

    kospi_ret = close["^KS11"] / close["^KS11"].iloc[0] - 1.0
    kosdaq_ret = close["^KQ11"] / close["^KQ11"].iloc[0] - 1.0
    blended = (kospi_ret * weights["KOSPI"] + kosdaq_ret * weights["KOSDAQ"]) * 100.0
    return pd.DataFrame({"날짜": close.index, "벤치마크누적수익률": blended.values})
