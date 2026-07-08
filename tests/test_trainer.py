"""
tests/test_trainer.py — retrain_daily 학습/서빙 파이프라인 일치 가드

2026-07-08 architect 검증에서 발견된 버그: retrain_daily()의 KR reversion
학습 블록이 FEATURE_COLS(17개, 레거시)로 dropna하고 train_global()에
feature_cols를 전달하지 않아, FEATURE_COLS_REVERSION(4개, 검증된 축소셋)이
분기별 자동 재학습에 전혀 반영되지 않았다 — 프로덕션 모델이 계속 17피처로
재생성되는 학습/서빙 스큐(train-serve skew) 버그.

이 테스트는 retrain_daily가 reversion 학습 시 반드시 FEATURE_COLS_REVERSION을
train_global에 전달하는지 고정한다(무거운 실제 다운로드/학습은 monkeypatch로 제거).
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

import ml.trainer as trainer_mod
import ml.features as features_mod
from ml.features import FEATURE_COLS_REVERSION

# test_scanner.py가 ml.model을 sys.modules에 스텁(mock)으로 setdefault 해두면
# (predict만 있고 train_global 없음) 이후 실행되는 이 테스트가 오염된 스텁을 보게 된다.
# ml.features는 conftest.py가 미리 보호하지만 ml.model은 그렇지 않으므로 여기서 직접 방어한다.
if not hasattr(sys.modules.get("ml.model"), "train_global"):
    sys.modules.pop("ml.model", None)
model_mod = importlib.import_module("ml.model")

N = 60  # HORIZON(10) 초과 + 슬라이스 후 10건 이상 남도록 충분히 크게


def _fake_ohlcv(n=N):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000.0},
        index=idx,
    )


@pytest.fixture
def capture_train_global(monkeypatch):
    """실제 다운로드/학습을 전부 목으로 대체하고 train_global 호출 인자를 캡처한다."""
    calls = []

    def _fake_train_global(combined_df, agent, feature_cols=None, wf_folds=None):
        calls.append({"agent": agent, "feature_cols": feature_cols})
        return None, {"auc": 0.6, "accuracy": 0.5}

    def _fake_add_features(raw_df, kospi_df=None):
        out = raw_df.copy()
        for col in FEATURE_COLS_REVERSION:
            out[col] = 0.0
        return out

    def _fake_triple_barrier(df, tp_pct, sl_pct, max_holding_days):
        return np.zeros(len(df)), np.zeros(len(df))

    def _fake_detect_reversion_rows(df):
        return pd.Series(True, index=df.index)

    monkeypatch.setattr(model_mod, "train_global", _fake_train_global)
    monkeypatch.setattr(features_mod, "add_features", _fake_add_features)
    monkeypatch.setattr(features_mod, "_triple_barrier_pnl", _fake_triple_barrier)
    monkeypatch.setattr(features_mod, "detect_reversion_rows", _fake_detect_reversion_rows)
    monkeypatch.setattr(trainer_mod, "fetch_3y", lambda ticker: _fake_ohlcv())
    monkeypatch.setattr(trainer_mod, "_fetch", lambda ticker, period: _fake_ohlcv())

    import signals.krx_universe as krx_mod
    monkeypatch.setattr(krx_mod, "get_krx_backtest_universe", lambda top_n=200: {"005930.KS": "삼성전자"})

    return calls


class TestRetrainDailyFeatureCols:
    def test_reversion_uses_feature_cols_reversion(self, capture_train_global):
        trainer_mod.retrain_daily(market="kr", period="3y")
        rev_calls = [c for c in capture_train_global if c["agent"] == "reversion"]
        assert len(rev_calls) == 1
        assert rev_calls[0]["feature_cols"] == FEATURE_COLS_REVERSION
