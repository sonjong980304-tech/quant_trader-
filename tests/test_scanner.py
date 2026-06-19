"""
tests/test_scanner.py
signals/scanner.py avg_win_effective 패널티 없음 검증 (레짐 필터 제거 후)
"""
import os
import sys
import types
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── 최소 의존성 모킹 ──────────────────────────────────────────────────────────

_mock_config = types.ModuleType("config")
_mock_config.ML_MIN_WIN_PROB    = 0.55
_mock_config.ML_MIN_RISK_REWARD = 1.5
sys.modules.setdefault("config", _mock_config)

_mock_features = types.ModuleType("ml.features")
_mock_features.compute_rsi = lambda s: pd.Series([50.0] * len(s), index=s.index)
_ml_pkg = types.ModuleType("ml")
sys.modules.setdefault("ml", _ml_pkg)
sys.modules.setdefault("ml.features", _mock_features)

_mock_model_mod = types.ModuleType("ml.model")
_mock_model_mod.predict = None  # monkeypatch로 교체
sys.modules.setdefault("ml.model", _mock_model_mod)

from signals.scanner import _eval_agent  # noqa: E402


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _dummy_df(n: int = 300) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    c = np.full(n, 10000.0)
    return pd.DataFrame(
        {"Open": c, "High": c, "Low": c, "Close": c, "Volume": np.full(n, 1_000_000)},
        index=dates,
    )


def _pred(win_prob=0.70, avg_win=0.15, avg_loss=0.03):
    return {
        "has_model": True,
        "win_prob": win_prob,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "model_auc": 0.65,
        "model_acc": 0.65,
    }


# ── avg_win_effective 패널티 없음 검증 ──────────────────────────────────────
# 레짐 필터(is_bear/adr_bear) 제거 후: avg_win_effective == avg_win (패널티 없음)

class TestNoRegimePenalty:
    def test_reversion_no_penalty(self, monkeypatch):
        """reversion — 레짐 무관하게 avg_win_effective == avg_win."""
        monkeypatch.setattr(sys.modules["ml.model"], "predict", lambda df, ticker, agent: _pred())
        result = _eval_agent(_dummy_df(), "T", "reversion")
        assert result is not None
        assert result["avg_win_effective"] == pytest.approx(0.15, rel=1e-4)

    def test_momentum_no_penalty(self, monkeypatch):
        """momentum — 레짐 무관하게 avg_win_effective == avg_win."""
        monkeypatch.setattr(sys.modules["ml.model"], "predict", lambda df, ticker, agent: _pred())
        result = _eval_agent(_dummy_df(), "T", "momentum")
        assert result is not None
        assert result["avg_win_effective"] == pytest.approx(0.15, rel=1e-4)
