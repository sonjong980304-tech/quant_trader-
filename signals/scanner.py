from __future__ import annotations

"""
scanner.py - 급등주 유니버스 스크리닝 + 기술적 트리거 탐지

흐름:
  1. 유니버스 필터 (유동성, 등락률)
  2. 기술적 트리거 (룰 기반 — OR 조건)
  3. ML 예측 (XGBoost 승률)
  4. 신호 조건: 승률 ≥ 55% AND 손익비 ≥ 1.5
"""

import logging
import datetime
import numpy as np
import pandas as pd
import pytz

from config import ML_MIN_WIN_PROB as MIN_WIN_PROB, ML_MIN_RISK_REWARD as MIN_RISK_REWARD

logger = logging.getLogger(__name__)
MIN_TRIGGERS    = 1      # 기술적 트리거 최소 1개 이상
MIN_MODEL_AUC   = 0.58   # 최소 모델 예측력 (0.5 = 동전던지기, 0.58+ = 참고 가능)

BEAR_AVG_WIN_PENALTY         = 0.4   # 약세장(MA/RSI) avg_win 보정 계수
BEAR_REVERSION_ADR_PENALTY   = 0.25  # ADR 약세장 시 reversion 전용 패널티

# 트리거 → 에이전트 매핑
_MOMENTUM_TRIGGERS  = {"거래량폭발", "BB스퀴즈돌파"}
_REVERSION_TRIGGERS = {"BB하단반등", "RSI과매도탈출", "이격도저점"}

# 구조적 하락 종목 블랙리스트 — 신호 발생 시 무시
BLACKLIST: set[str] = {"EL"}

KST = pytz.timezone("Asia/Seoul")

# 한국 장 시간 (분 단위)
_KR_OPEN_MIN  = 9 * 60       # 09:00
_KR_CLOSE_MIN = 15 * 60 + 30 # 15:30
_KR_TOTAL_MIN = _KR_CLOSE_MIN - _KR_OPEN_MIN  # 390분

# 미국 장 시간 (ET 기준, 분 단위)
_US_OPEN_MIN  = 9 * 60 + 30  # 09:30 ET
_US_CLOSE_MIN = 16 * 60      # 16:00 ET
_US_TOTAL_MIN = _US_CLOSE_MIN - _US_OPEN_MIN  # 390분


def _projected_volume(df: pd.DataFrame) -> float:
    """
    장중 거래량을 하루 예상 거래량으로 환산.
    한국장(KST 기준)과 미국장(ET 기준) 모두 지원.
    과거 데이터(백테스트)는 원본 그대로 반환.
    """
    last_vol  = float(df["Volume"].iloc[-1])
    last_date = df.index[-1]
    if hasattr(last_date, "date"):
        last_date = last_date.date()

    now_kst = datetime.datetime.now(KST)
    eastern = pytz.timezone("America/New_York")
    now_et  = now_kst.astimezone(eastern)

    # 한국장 체크
    if last_date == now_kst.date():
        kst_min = now_kst.hour * 60 + now_kst.minute
        if _KR_OPEN_MIN <= kst_min < _KR_CLOSE_MIN:
            elapsed = kst_min - _KR_OPEN_MIN
            if elapsed <= 0:
                return 0.0
            return last_vol / (elapsed / _KR_TOTAL_MIN)
        if kst_min >= _KR_CLOSE_MIN:
            return last_vol

    # 미국장 체크 (ET 기준)
    if last_date == now_et.date():
        et_min = now_et.hour * 60 + now_et.minute
        if _US_OPEN_MIN <= et_min < _US_CLOSE_MIN:
            elapsed = et_min - _US_OPEN_MIN
            if elapsed <= 0:
                return 0.0
            return last_vol / (elapsed / _US_TOTAL_MIN)
        if et_min >= _US_CLOSE_MIN:
            return last_vol

    return last_vol  # 과거 데이터 or 장 외 시간


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """ATR(14) 계산."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    prev  = close.shift(1)
    tr    = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr   = tr.rolling(period).mean()
    val   = float(atr.iloc[-1])
    return val if not np.isnan(val) else float(tr.iloc[-period:].mean())


# ─────────────────────────────────────────────
# 트리거 감지
# ─────────────────────────────────────────────

def detect_triggers(df: pd.DataFrame) -> list[str]:
    """
    기술적 트리거 탐지 (OR 조건 — 하나라도 충족 시 ML 모델 실행).

    5가지 트리거:
      ① 거래량폭발   : 예상 일거래량 > 20일 평균 × 2 + 양봉
      ② BB하단반등   : 전일 BB 하단 이탈 → 금일 재진입
      ③ RSI과매도탈출 : RSI 30 이하 → 30 돌파
      ④ 이격도저점   : EMA20 대비 -5% 이하
      ⑤ BB스퀴즈돌파 : BB 폭 60일 최저 이후 상단 돌파
    """
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

    if len(df) < 61:
        return []

    triggers = []
    last = df.iloc[-1]
    if isinstance(last, pd.DataFrame):
        last = last.iloc[-1]

    # ① 거래량폭발 (장중 시간 보정 적용)
    vol_ma20   = df["Volume"].rolling(20).mean().iloc[-2]
    proj_vol   = _projected_volume(df)
    if vol_ma20 > 0 and proj_vol > vol_ma20 * 2.0 and float(last["Close"]) > float(last["Open"]):
        triggers.append("거래량폭발")

    # ② BB하단반등
    bb_mid   = df["Close"].rolling(20).mean()
    bb_std   = df["Close"].rolling(20).std()
    bb_lower = bb_mid - 2 * bb_std
    if float(df["Close"].iloc[-2]) < float(bb_lower.iloc[-2]) and float(last["Close"]) >= float(bb_lower.iloc[-1]):
        triggers.append("BB하단반등")

    # ③ RSI 과매도 탈출
    from ml.features import compute_rsi
    rsi = compute_rsi(df["Close"])
    if float(rsi.iloc[-2]) < 30 and float(rsi.iloc[-1]) >= 30:
        triggers.append("RSI과매도탈출")

    # ④ 이격도 저점 (EMA20 대비 -5% 이하)
    ema20     = df["Close"].ewm(span=20, adjust=False).mean()
    deviation = (float(last["Close"]) - float(ema20.iloc[-1])) / float(ema20.iloc[-1])
    if deviation <= -0.05:
        triggers.append("이격도저점")

    # ⑤ BB 스퀴즈 돌파
    bb_upper    = bb_mid + 2 * bb_std
    bb_width    = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    min_width60 = bb_width.iloc[-61:-1].min()
    if (float(bb_width.iloc[-2]) <= min_width60 * 1.1
            and float(last["Close"]) > float(bb_upper.iloc[-1])):
        triggers.append("BB스퀴즈돌파")

    return triggers


# ─────────────────────────────────────────────
# 단일 종목 스캔
# ─────────────────────────────────────────────

def _eval_agent(df: pd.DataFrame, ticker: str, agent: str,
                is_bear: bool = False, adr_bear: bool = False) -> dict | None:
    """단일 에이전트 예측. 조건 미충족 시 None."""
    from ml.model import predict
    pred = predict(df, ticker, agent=agent)
    if not pred["has_model"]:
        logger.debug("  [%s][%s] 모델 없음 — 패스", ticker, agent)
        return None
    win_prob = pred.get("win_prob")
    avg_win  = pred.get("avg_win")
    avg_loss = pred.get("avg_loss")
    if win_prob is None or avg_win is None or avg_loss is None or avg_loss <= 0:
        return None
    model_auc = pred.get("model_auc", 0.0) or 0.0
    if model_auc < MIN_MODEL_AUC:
        logger.debug("  [%s][%s] AUC %.3f < %.2f — 패스", ticker, agent, model_auc, MIN_MODEL_AUC)
        return None
    if win_prob < MIN_WIN_PROB:
        logger.debug("  [%s][%s] 승률 %.1f%% < %.0f%% — 패스",
                     ticker, agent, win_prob * 100, MIN_WIN_PROB * 100)
        return None
    # 약세장 avg_win 보정
    # ADR 약세장 + reversion: 0.25 패널티 (지수 상승에도 개별 종목 하락 폭 큰 환경)
    # 그 외 약세장: 0.4 패널티
    if adr_bear and agent == "reversion":
        penalty = BEAR_REVERSION_ADR_PENALTY
    elif is_bear:
        penalty = BEAR_AVG_WIN_PENALTY
    else:
        penalty = 1.0
    effective_avg_win = avg_win * penalty
    expected_win  = effective_avg_win * win_prob
    expected_loss = avg_loss * (1 - win_prob)
    rr = expected_win / expected_loss if expected_loss > 0 else 0.0
    if rr < MIN_RISK_REWARD:
        logger.debug("  [%s][%s] 손익비 %.2f < %.1f — 패스", ticker, agent, rr, MIN_RISK_REWARD)
        return None
    return {**pred, "risk_reward": round(rr, 2), "agent": agent,
            "avg_win_effective": effective_avg_win}


def _is_uptrend(df: pd.DataFrame) -> bool:
    """MA200 추세 필터: 종가가 200일 이동평균 위에 있으면 True."""
    if len(df) < 200:
        return True  # 데이터 부족 시 필터 비활성화
    ma200 = df["Close"].rolling(200).mean().iloc[-1]
    return float(df["Close"].iloc[-1]) >= float(ma200)


def scan_ticker(ticker: str, df: pd.DataFrame, is_bear: bool = False) -> dict | None:
    """
    단일 종목에 대해 트리거 → 에이전트 분기 → 최종 에이전트 결합 → 신호 판단.

    에이전트 구조:
      돌파 에이전트   (momentum) : 거래량폭발, BB스퀴즈돌파 트리거 발생 시
      눌림목 에이전트 (reversion): BB하단반등, RSI과매도탈출, 이격도저점 트리거 발생 시
      최종 에이전트              : 두 에이전트 중 win_prob이 높은 쪽 선택
    """
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")].sort_index()

    # MA200 추세 필터: 하락 추세에서 매수 신호 차단
    if not _is_uptrend(df):
        logger.debug("  [%s] MA200 하락 추세 — 패스", ticker)
        return None

    triggers = detect_triggers(df)
    if not triggers:
        return None

    if len(triggers) < MIN_TRIGGERS:
        return None

    logger.info("  [%s] 트리거 감지: %s — ML 예측 실행", ticker, triggers)

    trigger_set   = set(triggers)
    has_momentum  = bool(trigger_set & _MOMENTUM_TRIGGERS)
    has_reversion = bool(trigger_set & _REVERSION_TRIGGERS)

    # ── 에이전트별 예측 ──────────────────────────────────────
    candidates = []
    if has_momentum:
        result = _eval_agent(df, ticker, "momentum", is_bear=is_bear)
        if result:
            candidates.append(result)
    if has_reversion:
        result = _eval_agent(df, ticker, "reversion", is_bear=is_bear)
        if result:
            candidates.append(result)

    # ── 최종 에이전트: win_prob 최고 선택 ────────────────────
    if not candidates:
        logger.debug("  [%s] 에이전트 조건 미충족 — 패스", ticker)
        return None

    best = max(candidates, key=lambda x: x["win_prob"])
    win_prob = best["win_prob"]
    avg_win  = best["avg_win"]
    avg_loss = best["avg_loss"]

    logger.info("  [%s] 신호 확정! 에이전트=%s 승률=%.1f%% 손익비=%.2f",
                ticker, best["agent"], win_prob * 100, best["risk_reward"])

    return {
        "ticker":        ticker,
        "triggers":      triggers,
        "agent":         best["agent"],
        "win_prob":      win_prob,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "risk_reward":   best["risk_reward"],
        "current_price": float(df["Close"].iloc[-1]),
        "model_acc":     best.get("model_acc"),
        "model_auc":     best.get("model_auc"),
        "atr":           _compute_atr(df),
    }


# ─────────────────────────────────────────────
# 전체 종목 스캔 (runner에서 호출)
# ─────────────────────────────────────────────

def scan_all(stocks: dict, fetch_fn, is_bear: bool = False) -> list[dict]:
    """
    관심종목 전체 스캔.

    stocks   : {ticker: name}
    fetch_fn : ticker → pd.DataFrame (OHLCV)
    is_bear  : 약세장 플래그 (avg_win 패널티 적용)

    반환: 신호 발생 종목 리스트
    """
    signals = []
    for ticker, name in stocks.items():
        if ticker in BLACKLIST:
            logger.debug("  [%s] 블랙리스트 — 스킵", ticker)
            continue
        try:
            df = fetch_fn(ticker)
            result = scan_ticker(ticker, df, is_bear=is_bear)
            if result:
                result["name"] = name
                signals.append(result)
        except Exception as e:
            logger.warning("  [%s] 스캔 실패: %s", ticker, e)
    return signals
