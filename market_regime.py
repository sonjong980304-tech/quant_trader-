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


def _notify_regime_change(prev: dict, is_blocked: bool, is_bear: bool, adr_bear: bool,
                          lc: float, adr: float | None) -> None:
    """상태 변화 시 텔레그램 알림 전송."""
    try:
        from notifier import send_telegram
    except Exception:
        return

    if is_blocked and not prev.get("is_blocked"):
        send_telegram(
            "🚨 <b>[시장 상황] 매수 전면 차단</b>\n"
            f"KOSPI {lc:,.0f} — MA5·MA20 역배열 확인\n"
            "신규 매수 신호를 모두 차단합니다."
        )
    elif is_bear and not prev.get("is_bear"):
        adr_str = f" (ADR {adr:.3f})" if adr is not None else ""
        send_telegram(
            "⚠️ <b>[시장 상황] 약세장 전략 적용 시작</b>\n"
            f"KOSPI {lc:,.0f}{adr_str}\n"
            "• MA/RSI 기반 약세장: avg_win × 0.4 패널티\n"
            "• ADR < 0.5: reversion avg_win × 0.25 패널티"
        )
    elif adr_bear and not prev.get("adr_bear"):
        adr_str = f"{adr:.3f}" if adr is not None else "N/A"
        send_telegram(
            "⚠️ <b>[시장 상황] ADR 약세장 감지</b>\n"
            f"ADR = {adr_str} (상승종목/하락종목 &lt; 0.5)\n"
            "reversion 에이전트 avg_win × 0.25 패널티 적용"
        )
    elif not is_bear and prev.get("is_bear"):
        send_telegram(
            "✅ <b>[시장 상황] 약세장 해제</b>\n"
            f"KOSPI {lc:,.0f} — 정상 시장으로 복귀\n"
            "패널티 없이 신호 탐색 재개"
        )


def get_market_regime() -> tuple[bool, bool, bool]:
    """폐기됨 — 레짐 필터 제거로 항상 (False, False, False) 반환.

    is_bear/adr_bear 기반 패널티 로직은 제거되었습니다.
    trend 에이전트 레짐 필터는 runner.py의 KOSPI MA200 검사로 대체됩니다.
    """
    return False, False, False
