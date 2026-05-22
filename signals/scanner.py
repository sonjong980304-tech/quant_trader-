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

logger = logging.getLogger(__name__)

MIN_WIN_PROB    = 0.55   # 최소 승률 55%
MIN_RISK_REWARD = 1.5    # 최소 손익비 1.5

KST = pytz.timezone("Asia/Seoul")

# 한국 장 시간 (분 단위)
_KR_OPEN_MIN  = 9 * 60       # 09:00
_KR_CLOSE_MIN = 15 * 60 + 30 # 15:30
_KR_TOTAL_MIN = _KR_CLOSE_MIN - _KR_OPEN_MIN  # 390분


def _projected_volume(df: pd.DataFrame) -> float:
    """
    장중 거래량을 하루 예상 거래량으로 환산.

    마지막 행이 오늘 날짜인 경우:
      - 9:00 이전: 0 반환 (신호 미발생)
      - 장중 (9:00~15:30): 현재 거래량 ÷ 경과비율
      - 장 종료 후: 원본 거래량 그대로

    마지막 행이 과거(백테스트)인 경우: 원본 그대로.
    """
    last_vol  = float(df["Volume"].iloc[-1])
    last_date = df.index[-1]
    if hasattr(last_date, "date"):
        last_date = last_date.date()

    today = datetime.date.today()
    if last_date != today:
        return last_vol  # 과거 데이터 or 장 종료 후 확정치

    now_kst   = datetime.datetime.now(KST)
    now_min   = now_kst.hour * 60 + now_kst.minute

    if now_min < _KR_OPEN_MIN:
        return 0.0  # 개장 전 — 신호 미발생

    elapsed   = now_min - _KR_OPEN_MIN
    if elapsed <= 0:
        return 0.0

    if now_min >= _KR_CLOSE_MIN:
        return last_vol  # 장 종료 — 확정 거래량

    elapsed_ratio = elapsed / _KR_TOTAL_MIN
    return last_vol / elapsed_ratio  # 하루 예상 거래량


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
    if len(df) < 61:
        return []

    triggers = []
    last = df.iloc[-1]

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

def scan_ticker(ticker: str, df: pd.DataFrame) -> dict | None:
    """
    단일 종목에 대해 트리거 → ML 예측 → 신호 판단.

    반환: 신호 dict (조건 미충족 시 None)
    신호 dict 키:
      ticker, name, triggers, win_prob, avg_win, avg_loss,
      risk_reward, current_price, kelly_fraction, kelly_amount
    """
    triggers = detect_triggers(df)
    if not triggers:
        return None

    logger.info("  [%s] 트리거 감지: %s — ML 예측 실행", ticker, triggers)

    from ml.model import predict
    pred = predict(df, ticker)

    if not pred["has_model"]:
        logger.debug("  [%s] 모델 없음 — 패스", ticker)
        return None

    win_prob = pred.get("win_prob")
    avg_win  = pred.get("avg_win")
    avg_loss = pred.get("avg_loss")

    if win_prob is None or avg_win is None or avg_loss is None or avg_loss <= 0:
        return None

    risk_reward = avg_win / avg_loss

    if win_prob < MIN_WIN_PROB:
        logger.debug("  [%s] 승률 %.1f%% < 55%% — 패스", ticker, win_prob * 100)
        return None

    if risk_reward < MIN_RISK_REWARD:
        logger.debug("  [%s] 손익비 %.2f < 1.5 — 패스", ticker, risk_reward)
        return None

    logger.info("  [%s] 🚨 신호 확정! 승률=%.1f%% 손익비=%.2f",
                ticker, win_prob * 100, risk_reward)

    return {
        "ticker":        ticker,
        "triggers":      triggers,
        "win_prob":      win_prob,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "risk_reward":   round(risk_reward, 2),
        "current_price": float(df["Close"].iloc[-1]),
        "model_acc":     pred.get("model_acc"),
        "model_auc":     pred.get("model_auc"),
    }


# ─────────────────────────────────────────────
# 전체 종목 스캔 (runner에서 호출)
# ─────────────────────────────────────────────

def scan_all(stocks: dict, fetch_fn) -> list[dict]:
    """
    관심종목 전체 스캔.

    stocks   : {ticker: name}
    fetch_fn : ticker → pd.DataFrame (OHLCV)

    반환: 신호 발생 종목 리스트
    """
    signals = []
    for ticker, name in stocks.items():
        try:
            df = fetch_fn(ticker)
            result = scan_ticker(ticker, df)
            if result:
                result["name"] = name
                signals.append(result)
        except Exception as e:
            logger.warning("  [%s] 스캔 실패: %s", ticker, e)
    return signals
