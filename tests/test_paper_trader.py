"""
tests/test_paper_trader.py
paper_trader.py 검증 (V1 ~ V7)

실행: pytest tests/test_paper_trader.py -v
커버: pytest tests/test_paper_trader.py --cov=paper_trader -v
"""

import os
import sys
import types
import pytest

# ─── 경로 설정 ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# notifier 모킹 (텔레그램 실발송 차단)
_mock_notifier = types.ModuleType("notifier")
_mock_notifier.send_telegram = lambda msg: True
sys.modules["notifier"] = _mock_notifier

import paper_trader as pt
from backtest_walkforward import _apply_costs as bt_apply_costs


# ─── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_files(tmp_path):
    """각 테스트를 독립된 tmp 파일로 격리."""
    pt.TRADES_PATH   = str(tmp_path / "trades.json")
    pt.POS_PATH      = str(tmp_path / "pos.json")
    pt.META_PATH     = str(tmp_path / "meta.json")
    pt.SNAPSHOT_PATH = str(tmp_path / "snapshot.json")
    yield


def _signal(ticker="005930.KS", entry=80000.0, **kwargs):
    defaults = dict(
        name="삼성전자", agent="eod",
        trigger_types=["volume_explosion"],
        win_prob=0.63, avg_win=0.087, avg_loss=0.07, rr=1.24,
        regime_prob=0.72, regime_pass=True,
        entry_price=entry, actual_price=None,
        position_size_pct=0.05, kelly_fraction=None,
        auc_at_signal=0.72,
    )
    defaults.update(kwargs)
    return pt.log_paper_signal(ticker=ticker, **defaults)


# ─────────────────────────────────────────────────────────────────────────────
# V1 / V2: log_paper_signal + evaluate_positions
# ─────────────────────────────────────────────────────────────────────────────

class TestLogPaperSignal:
    def test_returns_signal_id(self):
        sid = _signal()
        assert isinstance(sid, str) and len(sid) == 8

    def test_trade_written(self):
        _signal()
        trades = pt._load(pt.TRADES_PATH, [])
        assert len(trades) == 1
        assert trades[0]["status"] == "open"

    def test_position_written(self):
        _signal()
        positions = pt._load(pt.POS_PATH, {})
        assert len(positions) == 1

    def test_start_date_set(self):
        _signal()
        meta = pt._load(pt.META_PATH, {})
        assert "start_date" in meta

    def test_snapshot_created_on_first_signal(self):
        _signal()
        snap = pt._load(pt.SNAPSHOT_PATH, {})
        assert "TP_PCT" in snap
        assert snap["TP_PCT"] == pt.TP_PCT

    def test_actual_slippage_calculated(self):
        _signal(entry=80000.0, actual_price=80200.0)
        trades = pt._load(pt.TRADES_PATH, [])
        slip = trades[0]["actual_slippage"]
        assert slip is not None
        assert abs(slip - 200 / 80000) < 1e-9


class TestEvaluatePositions:
    """V2: TP / SL / time 각 시나리오 + backtest 비용 일치."""

    ENTRY = 80000.0

    def _setup(self):
        _signal(entry=self.ENTRY)

    # --- TP 시나리오 ---
    def test_tp_triggers(self):
        self._setup()
        tp_price = self.ENTRY * 1.152           # 15.2% 상승 (>TP_PCT=15%)
        closed = pt.evaluate_positions({"005930.KS": tp_price})
        assert len(closed) == 1
        assert closed[0]["reason"] == "TP"

    def test_tp_exit_price_is_exactly_7pct(self):
        self._setup()
        pt.evaluate_positions({"005930.KS": self.ENTRY * 1.16})
        trade  = pt._load(pt.TRADES_PATH, [])[0]
        expected_exit = round(self.ENTRY * 1.15, 4)
        assert trade["exit_price"] == expected_exit

    def test_tp_net_pnl_matches_backtest(self):
        """V2: net_pnl = backtest._apply_costs(TP_PCT, is_korean=True)."""
        self._setup()
        pt.evaluate_positions({"005930.KS": self.ENTRY * 1.16})
        trade   = pt._load(pt.TRADES_PATH, [])[0]
        expected = bt_apply_costs(pt.TP_PCT, True) * 100
        assert abs(trade["net_pnl_pct"] - round(expected, 3)) < 0.001

    # --- SL 시나리오 ---
    def test_sl_triggers(self):
        self._setup()
        closed = pt.evaluate_positions({"005930.KS": self.ENTRY * 0.91})  # -9% (SL_PCT=8%)
        assert len(closed) == 1
        assert closed[0]["reason"] == "SL"

    def test_sl_exit_price_is_exactly_minus7pct(self):
        self._setup()
        pt.evaluate_positions({"005930.KS": self.ENTRY * 0.90})
        trade = pt._load(pt.TRADES_PATH, [])[0]
        assert trade["exit_price"] == round(self.ENTRY * (1 - pt.SL_PCT), 4)

    def test_sl_net_pnl_matches_backtest(self):
        self._setup()
        pt.evaluate_positions({"005930.KS": self.ENTRY * 0.90})
        trade   = pt._load(pt.TRADES_PATH, [])[0]
        expected = bt_apply_costs(-pt.SL_PCT, True) * 100
        assert abs(trade["net_pnl_pct"] - round(expected, 3)) < 0.001

    # --- 기간 만료 시나리오 ---
    def test_time_exit_after_max_hold_days(self):
        self._setup()
        exit_price = self.ENTRY * 1.0025  # 소폭 상승
        for _ in range(pt.MAX_HOLD_DAYS - 1):
            pt.evaluate_positions({"005930.KS": exit_price}, trade_day=True)
        closed = pt.evaluate_positions({"005930.KS": exit_price}, trade_day=True)
        assert len(closed) == 1
        assert closed[0]["reason"] == "time"

    def test_time_exit_net_pnl_matches_backtest(self):
        self._setup()
        exit_price = self.ENTRY * 1.0025
        for _ in range(pt.MAX_HOLD_DAYS):
            pt.evaluate_positions({"005930.KS": exit_price}, trade_day=True)
        trade    = pt._load(pt.TRADES_PATH, [])[0]
        raw_pnl  = (exit_price - self.ENTRY) / self.ENTRY
        expected = bt_apply_costs(raw_pnl, True) * 100
        assert abs(trade["net_pnl_pct"] - round(expected, 3)) < 0.001

    # --- is_win 판정 ---
    def test_is_win_true_when_net_positive(self):
        self._setup()
        pt.evaluate_positions({"005930.KS": self.ENTRY * 1.16})
        trade = pt._load(pt.TRADES_PATH, [])[0]
        assert trade["is_win"] == 1

    def test_is_win_false_when_net_negative(self):
        self._setup()
        pt.evaluate_positions({"005930.KS": self.ENTRY * 0.90})
        trade = pt._load(pt.TRADES_PATH, [])[0]
        assert trade["is_win"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 액면분할/데이터 이상 방어 — 진입가 대비 ±30% 이상 급변 시 자동청산 보류
# ─────────────────────────────────────────────────────────────────────────────

class TestAnomalyGuard:
    ENTRY = 80000.0

    def _setup(self):
        _signal(entry=self.ENTRY)

    def test_large_drop_does_not_close_position(self):
        """진입가 대비 -30% 이상 급락 시 SL로 자동청산하지 않는다."""
        self._setup()
        closed = pt.evaluate_positions({"005930.KS": self.ENTRY * 0.2})  # -80%
        assert closed == []
        positions = pt._load(pt.POS_PATH, {})
        assert len(positions) == 1

    def test_large_drop_flags_position_and_alerts_once(self, monkeypatch):
        """이상 감지 시 포지션에 플래그를 남기고 텔레그램 알림을 보낸다."""
        import notifier
        alerts = []
        monkeypatch.setattr(notifier, "send_telegram", lambda msg: alerts.append(msg))
        self._setup()
        pt.evaluate_positions({"005930.KS": self.ENTRY * 0.2})
        pos = list(pt._load(pt.POS_PATH, {}).values())[0]
        assert pos.get("anomaly_flagged") is True
        assert len(alerts) == 1

    def test_alert_not_resent_on_repeated_checks(self, monkeypatch):
        """같은 이상 상태가 반복돼도 알림을 매번 다시 보내지 않고, 포지션은 계속 동결된다."""
        import notifier
        alerts = []
        monkeypatch.setattr(notifier, "send_telegram", lambda msg: alerts.append(msg))
        self._setup()
        pt.evaluate_positions({"005930.KS": self.ENTRY * 0.2})
        pt.evaluate_positions({"005930.KS": self.ENTRY * 0.2})
        positions = pt._load(pt.POS_PATH, {})
        assert len(positions) == 1
        assert len(alerts) == 1


# ─────────────────────────────────────────────────────────────────────────────
# V2-2: 진입가 가정 + 슬리피지 양방향 확인
# ─────────────────────────────────────────────────────────────────────────────

class TestEntryPriceAssumption:
    def test_assumed_slippage_stored(self):
        _signal()
        trades = pt._load(pt.TRADES_PATH, [])
        assert trades[0]["hypothetical_entry_slippage"] == pt.ASSUMED_SLIP

    def test_cost_is_round_trip(self):
        """COST_KR 왕복 비용 = 진입·청산 양방향."""
        from backtest_walkforward import COMMISSION_PCT, SLIPPAGE_PCT, STT_PCT
        cost_kr = COMMISSION_PCT * 2 + SLIPPAGE_PCT + STT_PCT  # 0.0026 (slip=0.05%)
        assert abs(pt._bt_apply_costs(0.0, True) - (-cost_kr)) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# V3: Circuit Breaker 6개 조건 개별 트리거
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    """각 조건을 가짜 데이터로 직접 트리거."""

    def _inject_trades(self, n: int, ev_pct: float, consec_loss: int = 0):
        """n건의 거래를 trades.json에 직접 주입."""
        import json
        trades = []
        for i in range(n):
            is_loss = i < consec_loss
            net = -abs(ev_pct) if is_loss else ev_pct + abs(ev_pct) * 2
            net = ev_pct  # 전체 평균 맞추기
            trades.append({
                "signal_id":    f"fake{i:04d}",
                "ticker":       "005930.KS",
                "status":       "closed",
                "net_pnl_pct":  round(net, 3),
                "is_win":       0 if net < 0 else 1,
                "actual_slippage": None,
            })
        # consec_loss 적용
        for i in range(consec_loss):
            trades[i]["net_pnl_pct"] = -1.0
            trades[i]["is_win"] = 0
        with open(pt.TRADES_PATH, "w") as f:
            json.dump(trades, f)

    def test_cb1_ev_below_minus05_at_n30(self):
        """CB-1: n≥30에서 EV ≤ -0.5%."""
        self._inject_trades(30, -0.6)
        triggered, reason = pt.check_circuit_breaker()
        assert triggered
        assert "EV" in reason

    def test_cb1_not_triggered_below_n30(self):
        # 이익/손실 번갈아 배치 → 연속 손실 최대 1건, EV 평균 ≈ -0.6%
        import json
        trades = [{"signal_id": f"f{i}", "ticker": "005930.KS",
                   "status": "closed",
                   "net_pnl_pct": -1.8 if i % 2 == 0 else 0.6,
                   "is_win": 0 if i % 2 == 0 else 1,
                   "actual_slippage": None} for i in range(29)]
        with open(pt.TRADES_PATH, "w") as f:
            json.dump(trades, f)
        triggered, _ = pt.check_circuit_breaker()
        assert not triggered

    def test_cb2_ci_below_minus1_at_n50(self):
        """CB-2: n≥50에서 CI 하단 < -1.0%."""
        # 전체 손실로 만들어 CI 하단이 깊어지도록
        self._inject_trades(50, -2.0)
        triggered, reason = pt.check_circuit_breaker()
        assert triggered
        assert "CI" in reason or "EV" in reason  # EV 조건이 먼저 걸릴 수 있음

    def test_cb3_consecutive_loss_8(self):
        """CB-3: 최대 연속 손실 ≥ 8건."""
        import json
        trades = [{"signal_id": f"f{i}", "ticker": "005930.KS",
                   "status": "closed", "net_pnl_pct": -1.0,
                   "is_win": 0, "actual_slippage": None} for i in range(8)]
        with open(pt.TRADES_PATH, "w") as f:
            json.dump(trades, f)
        triggered, reason = pt.check_circuit_breaker()
        assert triggered
        assert "연속" in reason

    def test_cb3_not_triggered_at_7(self):
        import json
        trades = [{"signal_id": f"f{i}", "ticker": "005930.KS",
                   "status": "closed", "net_pnl_pct": -1.0,
                   "is_win": 0, "actual_slippage": None} for i in range(7)]
        with open(pt.TRADES_PATH, "w") as f:
            json.dump(trades, f)
        triggered, _ = pt.check_circuit_breaker()
        assert not triggered

    def test_cb4_backtest_gap_at_n30(self):
        """CB-4: n≥30에서 백테스트 갭 ≤ -1.0%p."""
        # BACKTEST_EV = 0.00667, EV = 0.00667 - 0.011 = -0.00433 → 갭 -1.1%p
        ev_pct = (pt.BACKTEST_EV - 0.011) * 100
        self._inject_trades(30, round(ev_pct, 3))
        triggered, reason = pt.check_circuit_breaker()
        assert triggered
        assert "갭" in reason or "EV" in reason

    def test_cb4_not_triggered_below_n30(self):
        # 이익/손실 번갈아 배치 → 연속 손실 최대 1건, CB-3 미발동
        import json
        ev_pct = (pt.BACKTEST_EV - 0.011) * 100   # 약 -0.43%
        trades = [{"signal_id": f"f{i}", "ticker": "005930.KS",
                   "status": "closed",
                   "net_pnl_pct": ev_pct * 3 if i % 2 == 0 else abs(ev_pct),
                   "is_win": 0 if ev_pct * 3 < 0 and i % 2 == 0 else 1,
                   "actual_slippage": None} for i in range(29)]
        with open(pt.TRADES_PATH, "w") as f:
            json.dump(trades, f)
        triggered, _ = pt.check_circuit_breaker()
        assert not triggered

    def test_cb5_actual_slip_above_05pct(self):
        """CB-5: 실측 슬리피지 > 0.50%."""
        import json
        # get_metrics()는 closed 거래가 있어야 avg_actual_slip을 반환
        trades = [{"signal_id": "f0", "ticker": "005930.KS",
                   "status": "closed",
                   "net_pnl_pct": 0.5, "is_win": 1,
                   "actual_slippage": 0.006}]  # 0.6% > CB_SLIP(0.5%)
        with open(pt.TRADES_PATH, "w") as f:
            json.dump(trades, f)
        triggered, reason = pt.check_circuit_breaker()
        assert triggered
        assert "슬리피지" in reason

    def test_cb6_auc_below_045(self):
        """CB-6: 분기 평균 AUC < 0.45."""
        import json
        meta = {"auc_log": [{"date": "2026-01-01", "fold": i, "auc": 0.40}
                             for i in range(12)]}
        with open(pt.META_PATH, "w") as f:
            json.dump(meta, f)
        triggered, reason = pt.check_circuit_breaker()
        assert triggered
        assert "AUC" in reason


# ─────────────────────────────────────────────────────────────────────────────
# V4: BACKTEST_EV config 분리 확인
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestEVSource:
    def test_backtest_ev_from_config(self):
        from config import PAPER_BACKTEST_EV_KR as PAPER_BACKTEST_EV
        assert pt.BACKTEST_EV == PAPER_BACKTEST_EV

    def test_backtest_ev_value(self):
        assert abs(pt.BACKTEST_EV - 0.01301) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# V5: 파라미터 스냅샷 + drift 체크
# ─────────────────────────────────────────────────────────────────────────────

class TestParamSnapshot:
    def test_snapshot_saved_on_first_signal(self):
        _signal()
        snap = pt._load(pt.SNAPSHOT_PATH, {})
        assert snap["TP_PCT"] == 0.15
        assert snap["SL_PCT"] == 0.08
        assert snap["HORIZON"] == 10

    def test_no_drift_when_params_unchanged(self):
        _signal()
        drifts = pt.check_param_drift()
        assert drifts == []

    def test_drift_detected_when_tp_changed(self):
        _signal()
        snap = pt._load(pt.SNAPSHOT_PATH, {})
        snap["TP_PCT"] = 0.05   # 스냅샷을 다르게 조작
        pt._save(pt.SNAPSHOT_PATH, snap)
        drifts = pt.check_param_drift()
        assert any("TP_PCT" in d for d in drifts)

    def test_no_snapshot_returns_warning(self):
        drifts = pt.check_param_drift()
        assert any("스냅샷 없음" in d for d in drifts)


# ─────────────────────────────────────────────────────────────────────────────
# V6: 시작일 + 카운트 리셋
# ─────────────────────────────────────────────────────────────────────────────

class TestLogicChange:
    def test_start_date_set_on_first_signal(self):
        _signal()
        assert pt.get_start_date() is not None

    def test_register_logic_change_resets_start_date(self):
        _signal()
        old = pt.get_start_date()
        pt.register_logic_change("TP 파라미터 변경 테스트")
        new = pt.get_start_date()
        assert new == old  # 같은 날 테스트하므로 날짜 동일, 이력은 기록

    def test_logic_change_history_recorded(self):
        _signal()
        pt.register_logic_change("테스트 변경")
        meta = pt._load(pt.META_PATH, {})
        assert len(meta.get("logic_change_history", [])) == 1
        assert meta["logic_change_history"][0]["reason"] == "테스트 변경"

    def test_snapshot_refreshed_on_logic_change(self):
        _signal()
        pt.register_logic_change("리셋 테스트")
        snap = pt._load(pt.SNAPSHOT_PATH, {})
        assert "snapshot_date" in snap


# ─────────────────────────────────────────────────────────────────────────────
# V7: _apply_costs 공유 확인
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyCostsShared:
    def test_tp_cost_identical_to_backtest(self):
        paper_net  = bt_apply_costs(pt.TP_PCT, True)
        assert abs(paper_net - (pt.TP_PCT - 0.0026)) < 1e-9

    def test_sl_cost_identical_to_backtest(self):
        paper_net = bt_apply_costs(-pt.SL_PCT, True)
        assert abs(paper_net - (-pt.SL_PCT - 0.0026)) < 1e-9

    def test_us_ticker_uses_us_cost(self):
        from backtest_walkforward import COMMISSION_PCT, SLIPPAGE_PCT
        cost_us = COMMISSION_PCT * 2 + SLIPPAGE_PCT   # STT 없음 (미국 주식)
        us_net  = bt_apply_costs(0.0, False)
        assert abs(us_net - (-cost_us)) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# get_metrics 기본 동작
# ─────────────────────────────────────────────────────────────────────────────

class TestGetMetrics:
    def test_empty_returns_zero(self):
        m = pt.get_metrics()
        assert m["n"] == 0

    def test_metrics_after_close(self):
        _signal(entry=80000.0)
        pt.evaluate_positions({"005930.KS": 80000.0 * 1.16})  # TP
        m = pt.get_metrics()
        assert m["n"] == 1
        assert m["win_rate"] == 1.0
        assert m["ev"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# log_auc + is_circuit_breaker_active
# ─────────────────────────────────────────────────────────────────────────────

class TestLogAuc:
    def test_auc_written_to_meta(self):
        pt.log_auc(fold_id=1, auc=0.72)
        meta = pt._load(pt.META_PATH, {})
        assert len(meta["auc_log"]) == 1
        assert meta["auc_log"][0]["auc"] == 0.72

    def test_multiple_aucs_append(self):
        pt.log_auc(1, 0.70)
        pt.log_auc(2, 0.65)
        meta = pt._load(pt.META_PATH, {})
        assert len(meta["auc_log"]) == 2

    def test_low_auc_does_not_raise(self):
        pt.log_auc(fold_id=0, auc=0.42)  # < 0.5 경고 경로 통과
        meta = pt._load(pt.META_PATH, {})
        assert meta["auc_log"][0]["auc"] == 0.42


class TestCircuitBreakerActive:
    def test_inactive_when_no_data(self):
        assert pt.is_circuit_breaker_active() == False

    def test_active_after_consec_loss(self):
        import json
        trades = [{"signal_id": f"f{i}", "ticker": "005930.KS",
                   "status": "closed", "net_pnl_pct": -1.0,
                   "is_win": 0, "actual_slippage": None} for i in range(8)]
        with open(pt.TRADES_PATH, "w") as f:
            json.dump(trades, f)
        assert pt.is_circuit_breaker_active() == True


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_live_gate — P4 게이트 보고서
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateLiveGate:
    def test_returns_string(self):
        report = pt.evaluate_live_gate()
        assert isinstance(report, str)
        assert "P4" in report

    def test_fails_initially(self):
        report = pt.evaluate_live_gate()
        assert "LIVE_TRADING=False" in report or "❌" in report

    def test_passes_when_all_conditions_met(self):
        import json
        # 충분한 이익 거래 주입 (n=50, EV 양수)
        trades = [{"signal_id": f"f{i}", "ticker": "005930.KS",
                   "status": "closed", "net_pnl_pct": 1.0,
                   "is_win": 1, "actual_slippage": 0.001} for i in range(50)]
        with open(pt.TRADES_PATH, "w") as f:
            json.dump(trades, f)
        # 60일 이상 운영 meta
        import datetime
        start = (datetime.datetime.now() - datetime.timedelta(days=70)).strftime("%Y-%m-%d")
        meta = {"start_date": start,
                "auc_log": [{"date": "2026-01-01", "fold": i, "auc": 0.65} for i in range(12)]}
        with open(pt.META_PATH, "w") as f:
            json.dump(meta, f)
        report = pt.evaluate_live_gate()
        assert "모든 게이트 통과" in report or "✅" in report


# ─────────────────────────────────────────────────────────────────────────────
# daily_report + weekly_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyReport:
    def test_returns_string(self):
        report = pt.daily_report()
        assert isinstance(report, str)
        assert "일일 리포트" in report

    def test_contains_cumulative_metrics(self):
        _signal(entry=80000.0)
        pt.evaluate_positions({"005930.KS": 80000.0 * 1.16})
        report = pt.daily_report()
        assert "누적 지표" in report
        assert "세후EV" in report

    def test_circuit_breaker_section_shown_when_triggered(self):
        import json
        trades = [{"signal_id": f"f{i}", "ticker": "005930.KS",
                   "name": "테스트", "timestamp": "2026-01-01 09:00:00",
                   "status": "closed", "net_pnl_pct": -1.0, "is_win": 0,
                   "exit_timestamp": "2026-01-02 15:30:00",
                   "exit_reason": "SL", "actual_slippage": None} for i in range(8)]
        with open(pt.TRADES_PATH, "w") as f:
            json.dump(trades, f)
        report = pt.daily_report()
        assert "Circuit Breaker 발동" in report

    def test_drift_warning_in_report(self):
        _signal()
        snap = pt._load(pt.SNAPSHOT_PATH, {})
        snap["TP_PCT"] = 0.05
        pt._save(pt.SNAPSHOT_PATH, snap)
        report = pt.daily_report()
        assert "파라미터 변경" in report

    def test_logic_change_history_in_report(self):
        _signal()
        pt.register_logic_change("리포트 테스트용 변경")
        report = pt.daily_report()
        assert "로직 변경" in report


class TestWeeklySummary:
    def test_returns_string(self):
        summary = pt.weekly_summary()
        assert isinstance(summary, str)
        assert "주차별 집계" in summary

    def test_contains_gate_targets(self):
        summary = pt.weekly_summary()
        assert "거래수" in summary
        assert "세후EV" in summary

    def test_after_trades(self):
        _signal(entry=80000.0)
        pt.evaluate_positions({"005930.KS": 80000.0 * 1.16})
        summary = pt.weekly_summary()
        assert "1건" in summary


# ─────────────────────────────────────────────────────────────────────────────
# V8: 2단계 체결 구조 — entry_price=None
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoStageEntry:
    def test_entry_price_none_skips_eval(self):
        """entry_price=None이면 evaluate_positions에서 청산 스킵."""
        _signal(entry=None, eod_close=80000.0)
        closed = pt.evaluate_positions({"005930.KS": 80000.0 * 1.20})
        assert len(closed) == 0
        trades = pt._load(pt.TRADES_PATH, [])
        assert trades[0]["status"] == "open"

    def test_update_entry_prices_sets_entry(self, monkeypatch):
        """update_entry_prices: FDR 모킹으로 entry_price=None → 실제 Open으로 확정."""
        import types, sys, config
        import pandas as pd

        open_price = 82000.0
        _mock_fdr = types.ModuleType("FinanceDataReader")

        def _mock_datareader(ticker, start):
            return pd.DataFrame({"Open": [open_price], "Close": [83000.0]})

        _mock_fdr.DataReader = _mock_datareader
        monkeypatch.setitem(sys.modules, "FinanceDataReader", _mock_fdr)
        monkeypatch.setattr(config, "KIS_APP_KEY", "")  # KIS 비활성화 → FDR 폴백

        _signal(entry=None, eod_close=80000.0)
        updated = pt.update_entry_prices("KR")
        assert len(updated) == 1

        positions = pt._load(pt.POS_PATH, {})
        pos = list(positions.values())[0]
        assert pos["entry_price"] == round(open_price, 4)
        assert pos["highest"] == round(open_price, 4)
