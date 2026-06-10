"""
tests/test_position_manager.py
position_manager.py 검증
"""

import os
import sys
import types
import pytest
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── 외부 의존성 모킹 ───────────────────────────────────────────────────────

_mock_notifier = types.ModuleType("notifier")
_mock_notifier.send_telegram = lambda msg: True
sys.modules["notifier"] = _mock_notifier

_mock_trader = types.ModuleType("trader")
class _MockKISTrader:
    def get_balance(self): return []
    def get_current_price(self, code): return {"price": 0.0}
    def sell(self, code, qty): pass
    def sell_us(self, code, qty): pass
_mock_trader.KISTrader = _MockKISTrader
_mock_trader.positions = {}
sys.modules["trader"] = _mock_trader

_mock_pt = types.ModuleType("paper_trader")
_mock_pt.evaluate_positions_auto = lambda: None
sys.modules["paper_trader"] = _mock_pt

import position_manager as pm


# ─── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_STATE_FILE", str(tmp_path / "state.json"))
    yield


# ─── _load_state / _save_state ─────────────────────────────────────────────

class TestLoadSaveState:
    def test_load_default_when_missing(self):
        state = pm._load_state()
        assert state["bot_active"] is False
        assert state["legacy_tickers"] == []
        assert state["activated_at"] is None

    def test_roundtrip(self):
        data = {"bot_active": True, "legacy_tickers": ["005930"], "activated_at": "2026-01-01"}
        pm._save_state(data)
        assert pm._load_state() == data

    def test_ml_positions_preserved(self):
        pm._save_state({"bot_active": True, "ml_positions": {"A": {"qty": 3}}})
        assert pm._load_state()["ml_positions"]["A"]["qty"] == 3


# ─── is_bot_active ─────────────────────────────────────────────────────────

class TestIsBotActive:
    def test_false_by_default(self):
        assert pm.is_bot_active() is False

    def test_true_after_state_set(self):
        pm._save_state({"bot_active": True})
        assert pm.is_bot_active() is True


# ─── save_ml_position ──────────────────────────────────────────────────────

class TestSaveMlPosition:
    def test_position_stored(self):
        pm.save_ml_position("005930.KS", "삼성전자", qty=10, entry_price=70000.0, avg_win=0.15)
        pos = pm._load_state()["ml_positions"]["005930.KS"]
        assert pos["qty"] == 10
        assert pos["entry_price"] == 70000.0

    def test_target_price(self):
        pm.save_ml_position("A.KS", "종목A", qty=1, entry_price=10000.0, avg_win=0.15)
        pos = pm._load_state()["ml_positions"]["A.KS"]
        assert pos["target_price"] == pytest.approx(11500.0, rel=1e-4)

    def test_stop_price_no_atr(self):
        from config import SL_PCT
        pm.save_ml_position("B.KS", "종목B", qty=1, entry_price=10000.0, avg_win=0.15, atr=0.0)
        pos = pm._load_state()["ml_positions"]["B.KS"]
        assert pos["stop_price"] == pytest.approx(10000.0 * (1 - SL_PCT), rel=1e-4)

    def test_stop_price_with_atr(self):
        from config import SL_PCT
        # ATR=100, entry=10000 → stop = max(9800, 9400) = 9800
        pm.save_ml_position("C.KS", "종목C", qty=1, entry_price=10000.0, avg_win=0.15, atr=100.0)
        pos = pm._load_state()["ml_positions"]["C.KS"]
        expected = max(10000.0 - 2.0 * 100.0, 10000.0 * (1 - SL_PCT))
        assert pos["stop_price"] == pytest.approx(expected, rel=1e-4)

    def test_is_us_flag(self):
        pm.save_ml_position("AAPL", "Apple", qty=5, entry_price=200.0, avg_win=0.10, is_us=True)
        pos = pm._load_state()["ml_positions"]["AAPL"]
        assert pos["is_us"] is True

    def test_highest_price_initialized_to_entry(self):
        pm.save_ml_position("D.KS", "종목D", qty=1, entry_price=50000.0, avg_win=0.12)
        pos = pm._load_state()["ml_positions"]["D.KS"]
        assert pos["highest_price"] == pos["entry_price"]


# ─── _trading_days_elapsed ─────────────────────────────────────────────────

class TestTradingDaysElapsed:
    def test_today_returns_zero(self):
        assert pm._trading_days_elapsed(date.today().isoformat()) == 0

    def test_future_returns_zero(self):
        future = (date.today() + timedelta(days=5)).isoformat()
        assert pm._trading_days_elapsed(future) == 0

    def test_past_returns_positive(self):
        past = (date.today() - timedelta(days=10)).isoformat()
        assert pm._trading_days_elapsed(past) >= 1


# ─── check_ml_positions ────────────────────────────────────────────────────

def _make_pos(ticker, entry=10000.0, target=11500.0, stop=9400.0):
    return {
        "ticker": ticker, "name": "테스트", "qty": 1,
        "entry_price": entry, "target_price": target, "stop_price": stop,
        "entry_date": "2026-01-01", "is_us": False,
        "atr": 0.0, "highest_price": entry,
    }


class TestCheckMlPositions:
    def test_noop_when_empty(self):
        pm._save_state({"ml_positions": {}})
        pm.check_ml_positions()
        assert pm._load_state().get("ml_positions") == {}

    def test_tp_removes_position(self, monkeypatch):
        monkeypatch.setattr(pm, "_get_current_price", lambda t: 12000.0)
        monkeypatch.setattr(pm, "_trading_days_elapsed", lambda d: 0)
        pm._save_state({"ml_positions": {"T.KS": _make_pos("T.KS")}})
        pm.check_ml_positions()
        assert "T.KS" not in pm._load_state().get("ml_positions", {})

    def test_sl_removes_position(self, monkeypatch):
        monkeypatch.setattr(pm, "_get_current_price", lambda t: 9000.0)
        monkeypatch.setattr(pm, "_trading_days_elapsed", lambda d: 0)
        pm._save_state({"ml_positions": {"S.KS": _make_pos("S.KS")}})
        pm.check_ml_positions()
        assert "S.KS" not in pm._load_state().get("ml_positions", {})

    def test_horizon_removes_position(self, monkeypatch):
        monkeypatch.setattr(pm, "_get_current_price", lambda t: 10500.0)
        monkeypatch.setattr(pm, "_trading_days_elapsed", lambda d: 7)
        pm._save_state({"ml_positions": {"H.KS": _make_pos("H.KS")}})
        pm.check_ml_positions()
        assert "H.KS" not in pm._load_state().get("ml_positions", {})

    def test_no_trigger_keeps_position(self, monkeypatch):
        monkeypatch.setattr(pm, "_get_current_price", lambda t: 10500.0)
        monkeypatch.setattr(pm, "_trading_days_elapsed", lambda d: 3)
        pm._save_state({"ml_positions": {"K.KS": _make_pos("K.KS")}})
        pm.check_ml_positions()
        assert "K.KS" in pm._load_state().get("ml_positions", {})

    def test_highest_price_updated(self, monkeypatch):
        monkeypatch.setattr(pm, "_get_current_price", lambda t: 10800.0)
        monkeypatch.setattr(pm, "_trading_days_elapsed", lambda d: 3)
        pm._save_state({"ml_positions": {"U.KS": _make_pos("U.KS", entry=10000.0, target=11500.0)}})
        pm.check_ml_positions()
        pos = pm._load_state().get("ml_positions", {}).get("U.KS")
        assert pos is not None
        assert pos["highest_price"] == pytest.approx(10800.0, rel=1e-4)
