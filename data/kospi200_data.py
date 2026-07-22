from __future__ import annotations

"""
kospi200_data.py - KOSPI200 XGB 랭킹 전략 전용 데이터 레이어

기존 시총상위200 근사(signals/krx_universe.py)와 달리, pykrx 지수편입이력(1028=KOSPI200)으로
그 시점 실제 편입 200종목을 조회한다(생존편향 최소화). 투자자별 순매수·KOSPI 지수는
피처 계산(strategy/kospi200_agent.py)에서 원본 그대로 사용한다.

KOSPI200 편입이력은 pykrx 기준 2014-05-02부터 조회 가능(그 이전은 미제공).
"""

import time
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_INVESTOR_COL_MAP = {"기관합계": "Inst", "기타법인": "Corp", "개인": "Indiv", "외국인합계": "Foreign"}


def _ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def get_kospi200_pit(as_of=None, _sleep: float = 0.1) -> list[str]:
    """
    as_of 시점 KOSPI200 실제 편입 종목코드(6자리) 리스트. alternative=True로 휴장일 자동보정.
    as_of=None이면 오늘 기준.

    주의: pykrx.get_index_portfolio_deposit_file(ticker, date, alternative) — ticker가 먼저,
    date가 나중이다(순서를 바꾸면 date 문자열이 ticker로 해석되어 조용히 빈 결과가 나온다).
    """
    from pykrx import stock
    label = _ymd(as_of) if as_of is not None else _ymd(pd.Timestamp.now())
    lst = stock.get_index_portfolio_deposit_file("1028", label, alternative=True)
    time.sleep(_sleep)
    return [str(c).zfill(6) for c in lst]


def get_investor_net_buy(code: str, start, end, _sleep: float = 0.05) -> pd.DataFrame:
    """
    종목 일별 투자자유형 순매수대금(원). 컬럼: Inst, Corp, Indiv, Foreign.
    실패·데이터 없음 시 빈 DataFrame(같은 컬럼) 반환 — 호출부에서 NaN 중립 처리.
    """
    from pykrx import stock
    s, e = _ymd(start), _ymd(end)
    cols = ["Inst", "Corp", "Indiv", "Foreign"]
    try:
        raw = stock.get_market_trading_value_by_date(s, e, code)
    except Exception as ex:
        logger.warning("[KOSPI200] 투자자 순매수 조회 실패 %s: %s", code, ex)
        raw = pd.DataFrame()
    if raw is None or raw.empty:
        out = pd.DataFrame(columns=cols)
    else:
        out = raw.rename(columns=_INVESTOR_COL_MAP)
        out = out[[c for c in cols if c in out.columns]].copy()
        out.index = pd.DatetimeIndex(out.index)
    time.sleep(_sleep)
    return out
