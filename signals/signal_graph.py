"""
signal_graph.py - LangGraph 기반 신호 탐지 파이프라인

scan_ticker() 로직을 StateGraph 노드로 분리하여 각 단계를 명시적으로 표현.

그래프 구조:
  trigger_detect
    → (트리거 없음)        → END
    → (momentum만)        → momentum_only → select_best → END
    → (reversion만)       → reversion_only → select_best → END
    → (momentum+reversion) → both_agents → select_best → END
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict, Optional

from langgraph.graph import StateGraph, END

from signals.scanner import (
    detect_triggers,
    _is_uptrend,
    _eval_agent,
    _compute_atr,
    BLACKLIST,
    MIN_TRIGGERS,
    _MOMENTUM_TRIGGERS,
    _REVERSION_TRIGGERS,
)

logger = logging.getLogger(__name__)


class SignalState(TypedDict):
    ticker:           str
    name:             str
    df:               object          # pd.DataFrame
    is_bear:          bool
    triggers:         list[str]
    has_momentum:     bool
    has_reversion:    bool
    momentum_result:  Optional[dict]
    reversion_result: Optional[dict]
    final_signal:     Optional[dict]


# ── 노드 ─────────────────────────────────────────────────────────────────

def _node_trigger_detect(state: SignalState) -> SignalState:
    df     = state["df"]
    ticker = state["ticker"]

    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")].sort_index()

    if not _is_uptrend(df):
        logger.debug("  [%s] MA200 하락 추세 — 패스", ticker)
        return {**state, "df": df, "triggers": [], "has_momentum": False, "has_reversion": False}

    triggers    = detect_triggers(df)
    trigger_set = set(triggers)
    return {
        **state,
        "df":            df,
        "triggers":      triggers,
        "has_momentum":  bool(trigger_set & _MOMENTUM_TRIGGERS),
        "has_reversion": bool(trigger_set & _REVERSION_TRIGGERS),
    }


def _node_momentum_agent(state: SignalState) -> SignalState:
    result = _eval_agent(state["df"], state["ticker"], "momentum", is_bear=state["is_bear"])
    return {**state, "momentum_result": result}


def _node_reversion_agent(state: SignalState) -> SignalState:
    result = _eval_agent(state["df"], state["ticker"], "reversion", is_bear=state["is_bear"])
    return {**state, "reversion_result": result}


def _node_both_agents(state: SignalState) -> SignalState:
    """momentum + reversion 트리거가 동시에 발생한 경우 두 에이전트 모두 실행."""
    m = _eval_agent(state["df"], state["ticker"], "momentum",  is_bear=state["is_bear"])
    r = _eval_agent(state["df"], state["ticker"], "reversion", is_bear=state["is_bear"])
    return {**state, "momentum_result": m, "reversion_result": r}


def _node_select_best(state: SignalState) -> SignalState:
    ticker     = state["ticker"]
    candidates = [r for r in [state.get("momentum_result"), state.get("reversion_result")] if r]

    if not candidates:
        logger.debug("  [%s] 에이전트 조건 미충족 — 패스", ticker)
        return {**state, "final_signal": None}

    best = max(candidates, key=lambda x: x["win_prob"])
    df   = state["df"]

    logger.info("  [%s] 신호 확정! 에이전트=%s 승률=%.1f%% 손익비=%.2f",
                ticker, best["agent"], best["win_prob"] * 100, best["risk_reward"])

    return {
        **state,
        "final_signal": {
            "ticker":        ticker,
            "name":          state["name"],
            "triggers":      state["triggers"],
            "agent":         best["agent"],
            "win_prob":      best["win_prob"],
            "avg_win":       best["avg_win"],
            "avg_loss":      best["avg_loss"],
            "risk_reward":   best["risk_reward"],
            "current_price": float(df["Close"].iloc[-1]),
            "model_acc":     best.get("model_acc"),
            "model_auc":     best.get("model_auc"),
            "atr":           _compute_atr(df),
        },
    }


# ── 라우터 ────────────────────────────────────────────────────────────────

def _route_trigger(state: SignalState) -> str:
    triggers = state["triggers"]
    if not triggers or len(triggers) < MIN_TRIGGERS:
        return "no_trigger"
    if state["has_momentum"] and state["has_reversion"]:
        return "both"
    if state["has_momentum"]:
        return "momentum_only"
    return "reversion_only"


# ── 그래프 빌드 ───────────────────────────────────────────────────────────

def _build_signal_graph():
    g = StateGraph(SignalState)

    g.add_node("trigger_detect",  _node_trigger_detect)
    g.add_node("momentum_only",   _node_momentum_agent)
    g.add_node("reversion_only",  _node_reversion_agent)
    g.add_node("both_agents",     _node_both_agents)
    g.add_node("select_best",     _node_select_best)

    g.set_entry_point("trigger_detect")
    g.add_conditional_edges("trigger_detect", _route_trigger, {
        "no_trigger":     END,
        "momentum_only":  "momentum_only",
        "reversion_only": "reversion_only",
        "both":           "both_agents",
    })
    g.add_edge("momentum_only",  "select_best")
    g.add_edge("reversion_only", "select_best")
    g.add_edge("both_agents",    "select_best")
    g.add_edge("select_best",    END)

    return g.compile()


_signal_graph = _build_signal_graph()


# ── 공개 API ──────────────────────────────────────────────────────────────

def scan_ticker_graph(ticker: str, name: str, df, is_bear: bool = False) -> dict | None:
    """
    LangGraph 기반 단일 종목 신호 탐지.
    scanner.scan_ticker()와 동일한 결과를 반환하며 그래프 실행 이력을 추적 가능.
    """
    if ticker in BLACKLIST:
        return None
    try:
        result = _signal_graph.invoke({
            "ticker":           ticker,
            "name":             name,
            "df":               df,
            "is_bear":          is_bear,
            "triggers":         [],
            "has_momentum":     False,
            "has_reversion":    False,
            "momentum_result":  None,
            "reversion_result": None,
            "final_signal":     None,
        })
        return result.get("final_signal")
    except Exception as e:
        logger.warning("  [%s] signal_graph 스캔 실패: %s", ticker, e)
        return None


def scan_all_graph(stocks: dict, fetch_fn, is_bear: bool = False) -> list[dict]:
    """
    LangGraph 기반 전체 종목 스캔.
    runner.py의 scan_all() 대체 함수.
    """
    signals = []
    for ticker, name in stocks.items():
        if ticker in BLACKLIST:
            continue
        try:
            df     = fetch_fn(ticker)
            result = scan_ticker_graph(ticker, name, df, is_bear=is_bear)
            if result:
                signals.append(result)
        except Exception as e:
            logger.warning("  [%s] 스캔 실패: %s", ticker, e)
    return signals
