"""
kis_live.py - KIS 실시간 현재가 조회 (Streamlit 캐싱)

기존 trader.KISTrader 를 그대로 재사용한다(.env / config.IS_MOCK 자동 분기).
캐싱 정책:
  - 토큰/세션 : st.cache_resource 로 1회 발급 후 프로세스 내 재사용
  - 현재가     : st.cache_data(ttl=10) 로 10초 캐싱 → API 호출 폭주 방지
  - 조회 실패 : None 반환 → 호출부에서 직전 종가/진입가로 폴백
"""

import os
import sys

import streamlit as st

# 프로젝트 루트를 import 경로에 추가(trader/config 재사용)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def _get_trader():
    """
    KISTrader 1회 생성(토큰 발급 포함) 후 캐시.
    KIS 키가 없으면 None(라이브 비활성) 반환.
    """
    try:
        from config import KIS_APP_KEY
        if not KIS_APP_KEY:
            return None
        from trader import KISTrader
        return KISTrader()
    except Exception:
        return None


def is_live_available() -> bool:
    """KIS 라이브 조회 가능 여부."""
    return _get_trader() is not None


@st.cache_data(ttl=10, show_spinner=False)
def get_price(ticker: str):
    """
    현재가 조회(10초 캐싱). 국내(.KS/.KQ)·미국 자동 분기.
    실패 시 None.
    """
    t = _get_trader()
    if t is None:
        return None
    try:
        if ticker.endswith(".KS") or ticker.endswith(".KQ"):
            code = ticker.replace(".KS", "").replace(".KQ", "")
            return float(t.get_current_price(code)["price"])
        return float(t.get_us_current_price(ticker)["price"])
    except Exception:
        return None


def get_price_with_fallback(ticker: str, fallback):
    """현재가 우선, 실패하면 fallback(직전 종가/진입가)으로 대체."""
    p = get_price(ticker)
    return p if p is not None else fallback
