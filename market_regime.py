"""
KOSPI 기반 시장 상황 필터 (Market Regime Filter)

get_market_regime() 반환:
  is_blocked : 신규 BUY 전면 차단
               조건: KOSPI close < MA20 AND MA5 < MA20 (역배열)
  is_bear    : 약세장 플래그 (avg_win 패널티 적용)
               조건: KOSPI close < MA20 OR RSI(14) < 35
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_CACHE: dict = {}


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag    = gain.ewm(com=period - 1, min_periods=period).mean()
    al    = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = ag / al.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def get_market_regime() -> tuple[bool, bool]:
    """KOSPI 기반 시장 상황 판정 (일별 캐시). 반환: (is_blocked, is_bear)"""
    from datetime import date
    today_str = date.today().isoformat()
    if _CACHE.get("date") == today_str:
        return _CACHE["is_blocked"], _CACHE["is_bear"]
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader("KS11", start="2024-01-01")
        if df.empty or len(df) < 25:
            logger.warning("[MarketRegime] 데이터 부족 — 필터 비활성화")
            return False, False
        close = df["Close"]
        ma5   = close.rolling(5).mean()
        ma20  = close.rolling(20).mean()
        rsi14 = _rsi(close)
        lc, lm5, lm20 = float(close.iloc[-1]), float(ma5.iloc[-1]), float(ma20.iloc[-1])
        is_blocked = (lc < lm20) and (lm5 < lm20)
        is_bear    = (lc < lm20) or (rsi14 < 35)
        _CACHE.update({"date": today_str, "is_blocked": is_blocked, "is_bear": is_bear})
        logger.info("[MarketRegime] KOSPI=%.0f MA5=%.0f MA20=%.0f RSI14=%.1f → blocked=%s bear=%s",
                    lc, lm5, lm20, rsi14, is_blocked, is_bear)
        return is_blocked, is_bear
    except Exception as e:
        logger.warning("[MarketRegime] 조회 실패 — 비활성화: %s", e)
        return False, False
