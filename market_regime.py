"""
KOSPI 기반 시장 상황 필터 (Market Regime Filter)

get_market_regime() 반환: (is_blocked, is_bear, adr_bear)
  is_blocked : 신규 BUY 전면 차단
               조건: KOSPI close < MA20 AND MA5 < MA20 (역배열)
  is_bear    : 약세장 플래그 (avg_win 패널티 적용)
               조건: KOSPI close < MA20 OR RSI(14) < 35 OR adr_bear
  adr_bear   : ADR 기반 약세장 (reversion 패널티 0.25 적용)
               조건: KOSPI 상승종목 수 / 하락종목 수 < 0.5
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_CACHE: dict = {}
_ADR_CACHE: dict = {}


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag    = gain.ewm(com=period - 1, min_periods=period).mean()
    al    = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = ag / al.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def _get_adr_kospi() -> float | None:
    """
    KOSPI 전체 종목 ADR (상승종목 수 / 하락종목 수) 계산. 1시간 캐시.
    반환: ADR 값 (None이면 조회 실패)
    """
    from datetime import datetime as _dt
    cache_key = _dt.now().strftime("%Y-%m-%d %H")
    if _ADR_CACHE.get("key") == cache_key:
        return _ADR_CACHE.get("adr")
    try:
        import FinanceDataReader as fdr
        listing = fdr.StockListing("KOSPI")
        advances = int((listing["Changes"] > 0).sum())
        declines = int((listing["Changes"] < 0).sum())
        adr = advances / declines if declines > 0 else float("inf")
        _ADR_CACHE.update({"key": cache_key, "adr": adr})
        logger.info("[MarketRegime] ADR=%.3f (상승 %d / 하락 %d)", adr, advances, declines)
        return adr
    except Exception as e:
        logger.warning("[MarketRegime] ADR 조회 실패: %s", e)
        return None


def get_market_regime() -> tuple[bool, bool, bool]:
    """KOSPI 기반 시장 상황 판정 (5분 캐시). 반환: (is_blocked, is_bear, adr_bear)"""
    from datetime import datetime as _dt
    ts = _dt.now()
    cache_key = ts.strftime("%Y-%m-%d %H:") + f"{(ts.minute // 5) * 5:02d}"
    if _CACHE.get("key") == cache_key:
        return _CACHE["is_blocked"], _CACHE["is_bear"], _CACHE["adr_bear"]
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader("KS11", start="2024-01-01")
        if df.empty or len(df) < 25:
            logger.warning("[MarketRegime] 데이터 부족 — 필터 비활성화")
            return False, False, False
        close = df["Close"]
        ma5   = close.rolling(5).mean()
        ma20  = close.rolling(20).mean()
        rsi14 = _rsi(close)
        lc, lm5, lm20 = float(close.iloc[-1]), float(ma5.iloc[-1]), float(ma20.iloc[-1])
        is_blocked = (lc < lm20) and (lm5 < lm20)
        is_bear_ma = (lc < lm20) or (rsi14 < 35)

        adr      = _get_adr_kospi()
        adr_bear = (adr is not None) and (adr < 0.5)
        is_bear  = is_bear_ma or adr_bear

        _CACHE.update({"key": cache_key, "is_blocked": is_blocked,
                       "is_bear": is_bear, "adr_bear": adr_bear})
        logger.info(
            "[MarketRegime] KOSPI=%.0f MA5=%.0f MA20=%.0f RSI14=%.1f ADR=%s"
            " → blocked=%s bear=%s adr_bear=%s",
            lc, lm5, lm20, rsi14,
            f"{adr:.3f}" if adr is not None else "N/A",
            is_blocked, is_bear, adr_bear,
        )
        return is_blocked, is_bear, adr_bear
    except Exception as e:
        logger.warning("[MarketRegime] 조회 실패 — 비활성화: %s", e)
        return False, False, False
