from __future__ import annotations

"""
kospi200_agent.py - KOSPI200 XGB 랭킹 전략 (3번째 에이전트)

피처 3개 (quant_trader_backtest_dev/newstrat 백테스트로 검증된 정의 그대로):
  1) sector_momentum_zscore  : 20일 모멘텀(R20=close/close(-20)-1)을 같은날·같은섹터 내 z-score
  2) foreigner_institution_flow : (외국인+기관 순매수)/거래대금근사(Volume*Close)의 5일 이동평균
  3) kospi_realized_vol      : KOSPI 지수 20일 일별수익률 표준편차(원값, z 안 함)

라벨(학습 시): 7거래일 뒤 수익률의 그날 KOSPI200 편입종목 내 percentile rank(0~1).
매매: 예측 상위 10종목 균등매수 → 정확히 40거래일 보유 후 전량매도(TP/SL 없음).
      REBAL_FREQ=HOLD=40이므로 코호트가 겹치지 않는다 — "보유 포지션 0개일 때만 스캔"으로
      리밸런싱 주기를 자연스럽게 구현한다(별도 상태 파일 불필요, runner.py 쪽에서 게이팅).

그리드서치 근거(quant_trader_backtest_dev): HOLD=40/REBAL=40이 전체기간(2015~2026)과
2025년 상반기 절단 비교 양쪽에서 모두 최적(net 샤프 +0.480 / +0.461).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURES = ["sector_momentum_zscore", "foreigner_institution_flow", "kospi_realized_vol"]


def _tz_naive(idx: pd.Index) -> pd.Index:
    return idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx


def compute_features(
    codes: list[str],
    ohlcv_map: dict[str, pd.DataFrame],
    investor_map: dict[str, pd.DataFrame],
    kospi_realized_vol: float,
    sector_by_code: dict[str, str | None],
) -> pd.DataFrame:
    """종목코드 리스트의 '오늘' 스냅샷 피처를 계산해 code를 인덱스로 하는 DataFrame 반환."""
    rows = []
    for code in codes:
        df = ohlcv_map.get(code)
        if df is None or len(df) < 21:
            continue
        close = df["Close"].astype(float)
        close.index = _tz_naive(close.index)
        r20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)

        flow = np.nan
        iv = investor_map.get(code)
        if iv is not None and not iv.empty and {"Foreign", "Inst"}.issubset(iv.columns):
            iv = iv.copy()
            iv.index = _tz_naive(iv.index)
            fi = (iv["Foreign"] + iv["Inst"]).reindex(close.index)
            amount = (df["Volume"].astype(float) * close).replace(0, np.nan)
            flow_5dma = (fi / amount).rolling(5).mean()
            if len(flow_5dma) and pd.notna(flow_5dma.iloc[-1]):
                flow = float(flow_5dma.iloc[-1])

        rows.append({
            "code": code, "r20": r20,
            "foreigner_institution_flow": flow,
            "sector": sector_by_code.get(code),
        })

    feat = pd.DataFrame(rows)
    if feat.empty:
        return feat.set_index(pd.Index([], name="code"))
    feat = feat.set_index("code")

    # 섹터 내부 상대모멘텀 z-score — 같은 섹터 3종목 미만이면 무의미하므로 NaN → 중립(0) 처리
    g = feat.groupby("sector")["r20"]
    mu = g.transform("mean")
    sigma = g.transform("std").replace(0, np.nan)
    cnt = g.transform("count")
    z = (feat["r20"] - mu) / sigma
    feat["sector_momentum_zscore"] = z.where(cnt >= 3, np.nan).fillna(0.0)
    feat["foreigner_institution_flow"] = feat["foreigner_institution_flow"].fillna(0.0)
    feat["kospi_realized_vol"] = kospi_realized_vol

    return feat.dropna(subset=["kospi_realized_vol"])


def predict_ranking(feat: pd.DataFrame, model, top_n: int = 10) -> pd.DataFrame:
    """피처 DataFrame + 학습된 회귀모델로 예측점수 상위 top_n종목 반환(내림차순)."""
    X = feat[FEATURES].values.astype(np.float32)
    scored = feat.copy()
    scored["pred"] = model.predict(X)
    return scored.nlargest(top_n, "pred")


def _get_kospi_realized_vol(window: int = 20) -> float | None:
    """KOSPI 지수(^KS11, yfinance) 최근 window일 일별수익률 표준편차(가장 최근 값)."""
    try:
        import yfinance as yf
        df = yf.download("^KS11", period="3mo", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        rv = df["Close"].pct_change().rolling(window).std()
        val = rv.iloc[-1]
        return float(val) if pd.notna(val) else None
    except Exception as e:
        logger.warning("[KOSPI200] 실현변동성 계산 실패: %s", e)
        return None


def build_today_ranking(top_n: int = 10) -> pd.DataFrame | None:
    """
    오늘자 KOSPI200 PIT 유니버스를 스캔해 예측 상위 top_n종목을 반환한다.
    학습된 전역 모델(ml/models/_global_kospi200_xgb.pkl)이 없으면 None.
    """
    from ml.model import load_model
    model, _metrics = load_model("_global", "kospi200_xgb")
    if model is None:
        logger.warning("[KOSPI200] 학습된 모델 없음 — 스캔 스킵 (ml/kospi200_trainer 먼저 실행 필요)")
        return None

    from data.kospi200_data import get_kospi200_pit, get_investor_net_buy
    from data.data_fetcher import fetch_ohlcv
    from ml.trainer import _get_sector_map

    codes = get_kospi200_pit()
    if not codes:
        logger.warning("[KOSPI200] PIT 유니버스 조회 실패 — 스캔 스킵")
        return None

    sector_map = _get_sector_map(tickers=[f"{c}.KS" for c in codes])

    ohlcv_map: dict[str, pd.DataFrame] = {}
    for code in codes:
        try:
            df = fetch_ohlcv(f"{code}.KS", period_years=1)
            if df is not None and len(df) >= 25:
                ohlcv_map[code] = df
        except Exception as e:
            logger.debug("[KOSPI200] OHLCV 실패 %s: %s", code, e)
    if not ohlcv_map:
        logger.warning("[KOSPI200] OHLCV 조회 결과 0개 — 스캔 스킵")
        return None

    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=20)
    investor_map = {code: get_investor_net_buy(code, start, end) for code in ohlcv_map}

    kospi_rv = _get_kospi_realized_vol()
    if kospi_rv is None:
        logger.warning("[KOSPI200] KOSPI 실현변동성 계산 실패 — 스캔 스킵")
        return None

    sector_by_code = {c: sector_map.get(f"{c}.KS") for c in ohlcv_map}
    feat = compute_features(list(ohlcv_map.keys()), ohlcv_map, investor_map, kospi_rv, sector_by_code)
    if feat.empty:
        logger.warning("[KOSPI200] 피처 계산 결과 0행 — 스캔 스킵")
        return None

    return predict_ranking(feat, model, top_n=top_n)
