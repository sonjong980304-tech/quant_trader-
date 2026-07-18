"""
data_loader.py - 기존 봇 데이터(JSON/CSV) 읽기 전용 로더

봇 코드는 절대 수정하지 않고, 봇이 생성한 산출물 파일만 읽는다.
  - paper_trades.json     : 페이퍼 신호/거래 기록 (list)
  - paper_positions.json  : 페이퍼 오픈 포지션 (dict)
  - paper_meta.json       : 페이퍼 시작일 등
  - trade_history.csv     : 실매매 매수/매도 이력 (CSV, utf-8-sig)
  - state.json            : 실매매 보유 포지션 ml_positions
모델 AUC는 ml.model.load_model 로 읽는다 (G1: 페이퍼에 AUC 미저장 대응).
"""

import os
import sys
import json

import numpy as np
import pandas as pd

# ── 프로젝트 루트(= dashboard/의 부모)를 import 경로에 추가 ──────────────
# 봇 모듈(ml.model 등)을 읽기 위해 필요. 데이터 파일 경로도 루트 기준으로 잡는다.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── 데이터 파일 경로 (봇과 동일 위치) ────────────────────────────────────
PAPER_TRADES_PATH = os.path.join(_PROJECT_ROOT, "paper_trades.json")
PAPER_POS_PATH    = os.path.join(_PROJECT_ROOT, "paper_positions.json")
PAPER_META_PATH   = os.path.join(_PROJECT_ROOT, "paper_meta.json")
TRADE_HISTORY_CSV = os.path.join(_PROJECT_ROOT, "trade_history.csv")
STATE_JSON        = os.path.join(_PROJECT_ROOT, "state.json")

# 페이퍼 수치 컬럼 (문자→숫자 변환 대상)
_PAPER_NUM_COLS = [
    "win_prob", "rr", "avg_win", "avg_loss",
    "raw_pnl_pct", "net_pnl_pct", "holding_days",
    "hypothetical_entry_price",
]
# CSV 수치 컬럼
_CSV_NUM_COLS = [
    "entry_price", "exit_price", "qty", "pnl_amount", "pnl_pct",
    "win_prob", "avg_win_pct", "avg_loss_pct", "model_auc",
]


# ─────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────
def _read_json(path: str, default):
    """JSON 안전 로드. 없거나 손상되면 default 반환(타입 가드 포함)."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if type(data) is type(default) else default
    except Exception:
        return default


def _to_num(df: pd.DataFrame, cols: list):
    """지정 컬럼을 숫자형으로 변환(실패 시 NaN)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────
# 페이퍼 트레이딩
# ─────────────────────────────────────────────────────────────────────────
def load_paper_trades() -> pd.DataFrame:
    """
    paper_trades.json → DataFrame.
    - trigger_types(list) 를 'trigger_str'(문자열)로 펼쳐 보조 컬럼 추가
    - 오픈 포지션의 실제 진입가(시초가 확정값)를 '진입가'로 병합
    """
    rows = _read_json(PAPER_TRADES_PATH, [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = _to_num(df, _PAPER_NUM_COLS)

    # 트리거 리스트 → 문자열
    if "trigger_types" in df.columns:
        df["trigger_str"] = df["trigger_types"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else ""
        )
    else:
        df["trigger_types"] = [[] for _ in range(len(df))]
        df["trigger_str"] = ""

    # 오픈 포지션의 실제 진입가(시초가) 병합 → 없으면 가설 진입가로 폴백
    pos = _read_json(PAPER_POS_PATH, {})
    pos_entry = {
        v.get("signal_id"): v.get("entry_price")
        for v in pos.values()
    } if isinstance(pos, dict) else {}

    def _entry(r):
        e = pos_entry.get(r.get("signal_id"))
        if e is not None:
            return e
        return r.get("hypothetical_entry_price")

    df["진입가"] = df.apply(_entry, axis=1)
    return df


def load_paper_positions() -> pd.DataFrame:
    """paper_positions.json(dict) → DataFrame(오픈 포지션)."""
    pos = _read_json(PAPER_POS_PATH, {})
    if not isinstance(pos, dict) or not pos:
        return pd.DataFrame()
    return pd.DataFrame(list(pos.values()))


def get_paper_start_date():
    """페이퍼 시작일(없으면 None)."""
    meta = _read_json(PAPER_META_PATH, {})
    return meta.get("start_date") if isinstance(meta, dict) else None


def paper_summary(df: pd.DataFrame) -> dict:
    """페이퍼 요약 카드용 지표."""
    if df.empty:
        return {"총신호": 0, "평균승률": None, "누적수익률": None, "진행중": 0}
    closed = df[df["status"] == "closed"]
    open_n = int((df["status"] == "open").sum())

    win_rate = None
    if not closed.empty and "is_win" in closed.columns:
        wins = closed["is_win"].dropna()
        if len(wins):
            win_rate = float(pd.Series(wins).astype(bool).mean()) * 100

    cum = None
    if not closed.empty and "net_pnl_pct" in closed.columns:
        s = closed["net_pnl_pct"].dropna()
        if len(s):
            cum = float(s.sum())

    return {
        "총신호": int(len(df)),
        "평균승률": win_rate,
        "누적수익률": cum,
        "진행중": open_n,
    }


STRATEGY_CUTOFF = "2026-07-09 00:01:29"  # reversion 피처 12→4 축소(커밋 b4d75bf) 시점


def split_by_cutoff(df: pd.DataFrame, cutoff: str = STRATEGY_CUTOFF):
    """timestamp 기준 (현재 전략 df, 과거 전략 df) 튜플로 분리.

    cutoff 이후 = 현재 운용 전략(reversion 4피처), 이전 = 구버전(12피처) 참고용 기록.
    """
    if df.empty:
        return df, df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    cutoff_ts = pd.to_datetime(cutoff)
    current = df[ts >= cutoff_ts].reset_index(drop=True)
    past = df[ts < cutoff_ts].reset_index(drop=True)
    return current, past


def paper_ev_stats(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> dict:
    """
    청산 거래의 기댓값(EV, 거래당 평균 net_pnl_pct)과 부트스트랩 95% 신뢰구간.
    paper_trader.py의 get_metrics()와 동일한 방식(복원추출 2000회, seed=42)으로
    계산해 봇의 공식 판단 지표와 일관성을 맞춘다.
    """
    if df.empty:
        return {"n": 0, "ev": None, "ci_low": None, "ci_high": None}
    closed = df[df["status"] == "closed"]
    closed = closed[closed["net_pnl_pct"].notna()]
    n = len(closed)
    if n == 0:
        return {"n": 0, "ev": None, "ci_low": None, "ci_high": None}

    pnl = closed["net_pnl_pct"].to_numpy() / 100.0
    ev = float(pnl.mean())
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(pnl, size=n, replace=True).mean() for _ in range(n_boot)])
    ci_low, ci_high = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {
        "n": n,
        "ev": ev * 100,
        "ci_low": ci_low * 100,
        "ci_high": ci_high * 100,
    }


def paper_ev_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    청산 거래를 청산시각 순으로 정렬해 러닝 EV(그 시점까지의 누적 평균 net_pnl_pct)
    곡선 산출. 단순 합산(cumsum)이 아니라 거래당 평균이라 paper_ev_stats()의
    최종 EV와 같은 단위 — 거래가 늘어날수록 EV가 어느 값으로 수렴하는지 보여준다.
    """
    if df.empty:
        return pd.DataFrame()
    closed = df[df["status"] == "closed"].copy()
    closed = closed[closed["net_pnl_pct"].notna()]
    if closed.empty:
        return pd.DataFrame()
    closed["청산시각"] = pd.to_datetime(closed["exit_timestamp"], errors="coerce")
    closed = closed.sort_values("청산시각")
    closed["EV"] = closed["net_pnl_pct"].expanding().mean()
    return closed[["청산시각", "name", "ticker", "net_pnl_pct", "EV"]]


_AGENTS = ["trend", "reversion"]


def paper_agent_perf(df: pd.DataFrame) -> pd.DataFrame:
    """
    에이전트(trend/reversion)별 성과: 평균수익률·승률·거래수.
    거래 기록이 아직 없는 에이전트(예: trend 슬롯 미체결)도 0으로 채워
    항상 두 에이전트가 함께 표시되게 한다.
    """
    base = pd.DataFrame({"에이전트": _AGENTS})
    closed = df[df["status"] == "closed"].copy() if not df.empty else df
    closed = closed[closed["net_pnl_pct"].notna()] if not closed.empty else closed
    if closed.empty:
        base["평균수익률"] = 0.0
        base["거래수"] = 0
        base["승률"] = None
        return base

    g = closed.groupby("agent").agg(
        평균수익률=("net_pnl_pct", "mean"),
        거래수=("signal_id", "count"),
    ).reset_index()
    # 승률(is_win)은 별도 계산(None 제외)
    wr = (
        closed.dropna(subset=["is_win"])
        .groupby("agent")["is_win"]
        .apply(lambda s: pd.Series(s).astype(bool).mean() * 100)
        .reset_index(name="승률")
    )
    g = g.merge(wr, on="agent", how="left").rename(columns={"agent": "에이전트"})
    g = base.merge(g, on="에이전트", how="left")
    g["평균수익률"] = g["평균수익률"].fillna(0.0)
    g["거래수"] = g["거래수"].fillna(0).astype(int)
    return g


_TRIGGERS = ["거래량폭발", "BB하단반등", "RSI과매도탈출", "이격도저점", "BB스퀴즈돌파"]


def paper_trigger_perf(df: pd.DataFrame) -> pd.DataFrame:
    """
    트리거(reversion 5종 고정 + 실데이터에만 있는 그 외, 예: trend_entry)별
    평균수익률·거래수. 아직 한 번도 안 나온 트리거도 0으로 채워 항상 표시.
    """
    base = pd.DataFrame({"트리거": _TRIGGERS})
    closed = df[df["status"] == "closed"].copy() if not df.empty else df
    closed = closed[closed["net_pnl_pct"].notna()] if not closed.empty else closed
    if closed.empty or "trigger_types" not in closed.columns:
        base["평균수익률"] = 0.0
        base["거래수"] = 0
        return base

    ex = closed.explode("trigger_types")
    ex = ex[ex["trigger_types"].notna() & (ex["trigger_types"] != "")]
    if ex.empty:
        base["평균수익률"] = 0.0
        base["거래수"] = 0
        return base

    g = ex.groupby("trigger_types").agg(
        평균수익률=("net_pnl_pct", "mean"),
        거래수=("signal_id", "count"),
    ).reset_index().rename(columns={"trigger_types": "트리거"})
    extra = g[~g["트리거"].isin(_TRIGGERS)]
    g = base.merge(g, on="트리거", how="left")
    g["평균수익률"] = g["평균수익률"].fillna(0.0)
    g["거래수"] = g["거래수"].fillna(0).astype(int)
    return pd.concat([g, extra], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────
# 모델 AUC (G1: 페이퍼 auc_at_signal 미저장 → 모델 메타에서 조회)
# ─────────────────────────────────────────────────────────────────────────
_AUC_CACHE = {}


def get_model_auc(agent: str = "reversion"):
    """저장된 _global 모델의 AUC(마지막 fold). 실패 시 None. (간단 캐시)"""
    if agent in _AUC_CACHE:
        return _AUC_CACHE[agent]
    auc = None
    try:
        from ml.model import load_model
        _, m = load_model("_global", agent)
        auc = m.get("auc")
    except Exception:
        auc = None
    _AUC_CACHE[agent] = auc
    return auc


# ─────────────────────────────────────────────────────────────────────────
# 실제 매매
# ─────────────────────────────────────────────────────────────────────────
def load_trade_history() -> pd.DataFrame:
    """trade_history.csv → DataFrame (실매매 매수/매도 이력)."""
    if not os.path.exists(TRADE_HISTORY_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(TRADE_HISTORY_CSV, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    return _to_num(df, _CSV_NUM_COLS)


def load_live_positions() -> pd.DataFrame:
    """state.json 의 ml_positions(보유 포지션) → DataFrame. 없으면 빈 DF."""
    state = _read_json(STATE_JSON, {})
    ml = state.get("ml_positions", {}) if isinstance(state, dict) else {}
    if not isinstance(ml, dict) or not ml:
        return pd.DataFrame()
    return pd.DataFrame(list(ml.values()))


def _closed_history(df_hist: pd.DataFrame) -> pd.DataFrame:
    """청산 완료(exit_date 채워진) 거래만 필터."""
    if df_hist.empty or "exit_date" not in df_hist.columns:
        return pd.DataFrame()
    closed = df_hist[
        df_hist["exit_date"].notna() & (df_hist["exit_date"].astype(str) != "")
    ].copy()
    return closed


def live_summary(df_hist: pd.DataFrame, df_pos: pd.DataFrame) -> dict:
    """실매매 요약 카드용 지표."""
    closed = _closed_history(df_hist)
    realized = float(closed["pnl_amount"].dropna().sum()) if not closed.empty else 0.0
    win_rate = None
    if not closed.empty and "win" in closed.columns:
        w = closed["win"].dropna()
        if len(w):
            win_rate = float(pd.to_numeric(w, errors="coerce").fillna(0).mean()) * 100
    return {
        "보유종목": int(len(df_pos)),
        "실현손익": realized,
        "승률": win_rate,
        "거래수": int(len(closed)),
    }


def live_realized_curve(df_hist: pd.DataFrame) -> pd.DataFrame:
    """청산일 순 누적 실현손익(원) 곡선."""
    closed = _closed_history(df_hist)
    if closed.empty or "pnl_amount" not in closed.columns:
        return pd.DataFrame()
    closed["청산일"] = pd.to_datetime(closed["exit_date"], errors="coerce")
    closed = closed.sort_values("청산일")
    closed["누적실현손익"] = closed["pnl_amount"].fillna(0).cumsum()
    return closed[["청산일", "name", "ticker", "pnl_amount", "누적실현손익"]]


def live_strategy_perf(df_hist: pd.DataFrame) -> pd.DataFrame:
    """
    전략별 실현손익 (G2: CSV에 agent 컬럼이 없어 strategy 로 그룹핑).
    """
    closed = _closed_history(df_hist)
    if closed.empty or "strategy" not in closed.columns:
        return pd.DataFrame()
    g = closed.groupby("strategy").agg(
        실현손익합=("pnl_amount", "sum"),
        평균수익률=("pnl_pct", "mean"),
        거래수=("trade_id", "count"),
    ).reset_index().rename(columns={"strategy": "전략"})
    return g
