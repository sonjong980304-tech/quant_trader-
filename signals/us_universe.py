from __future__ import annotations

"""
us_universe.py - S&P 500 전체 종목 1차 스크리닝

yfinance로 S&P 500 종목 최근 22일 데이터를 일괄 다운로드 후
오늘 등락률 > 0% + 거래량 비율 상위 top_n종목 반환.
KRX의 pykrx 역할을 미국장에서 수행.
"""

import io
import logging
import datetime
import requests
import pandas as pd
import yfinance as yf
import pytz

logger = logging.getLogger(__name__)

_US_OPEN_MIN  = 9 * 60 + 30   # 09:30 ET
_US_CLOSE_MIN = 16 * 60        # 16:00 ET
_US_TOTAL_MIN = _US_CLOSE_MIN - _US_OPEN_MIN


def _project_us_volume(current_vol: float) -> float:
    """미국 장 시간(ET) 기준 현재 거래량을 하루 예상 거래량으로 환산."""
    eastern = pytz.timezone("America/New_York")
    now_et  = datetime.datetime.now(eastern)
    et_min  = now_et.hour * 60 + now_et.minute

    if et_min < _US_OPEN_MIN:
        return 0.0
    if et_min >= _US_CLOSE_MIN:
        return current_vol

    elapsed = et_min - _US_OPEN_MIN
    if elapsed <= 0:
        return 0.0
    return current_vol / (elapsed / _US_TOTAL_MIN)


def _get_sp500_tickers() -> list[str]:
    """Wikipedia에서 S&P 500 티커 목록 조회 (User-Agent 설정으로 403 우회)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        df = pd.read_html(io.StringIO(resp.text))[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info("S&P 500 목록 조회 완료: %d종목", len(tickers))
        return tickers
    except Exception as e:
        logger.warning("S&P 500 목록 조회 실패: %s", e)
        return []


def get_us_candidates(top_n: int = 50) -> dict[str, str]:
    """
    오늘 S&P 500 기준 급등 후보 종목 반환.

    필터 기준:
      1. 오늘 등락률 > 0%
      2. 오늘 거래량 > 직전 20일 평균 × 1.5
      3. 거래량 비율 상위 top_n개

    반환: {ticker: ticker}  (이름 미제공 — ticker로 대체)
    """
    tickers = _get_sp500_tickers()
    if not tickers:
        return {}

    logger.info("S&P 500 %d종목 22일 배치 다운로드 중...", len(tickers))
    try:
        raw = yf.download(
            tickers,
            period="22d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        logger.warning("yfinance 배치 다운로드 실패: %s", e)
        return {}

    vol_ratios: dict[str, float] = {}

    for ticker in tickers:
        try:
            df = raw[ticker].copy() if len(tickers) > 1 else raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Close", "Volume"]].dropna()
            if len(df) < 3:
                continue

            today_ret = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]
            if today_ret <= 0:
                continue

            vol_ma = df["Volume"].iloc[:-1].mean()
            if vol_ma <= 0:
                continue

            proj_vol  = _project_us_volume(float(df["Volume"].iloc[-1]))
            vol_ratio = proj_vol / vol_ma
            if vol_ratio < 1.5:
                continue

            vol_ratios[ticker] = vol_ratio
        except Exception:
            continue

    top = sorted(vol_ratios, key=lambda t: vol_ratios[t], reverse=True)[:top_n]
    candidates = {t: t for t in top}

    logger.info("US 후보 종목: %d개 (S&P 500 필터)", len(candidates))
    return candidates
