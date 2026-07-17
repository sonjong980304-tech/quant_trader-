"""
tests/test_nb_scorer.py — news_briefing.scorer 채점 로직 검증

db(get_pending_forecasts/update_forecast_verdict/get_hit_rate)와 market_data_fn은
전부 mock/주입한다. sleep_fn도 mock으로 대체해 재시도 테스트가 실제로 대기하지 않게 한다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from news_briefing import scorer  # noqa: E402


def _forecast(fid, market, direction):
    """db.get_pending_forecasts가 반환하는 형태의 dict를 흉내낸다."""
    return {
        "id": fid,
        "briefing_id": 1,
        "market": market,
        "direction": direction,
        "rationale": "근거",
        "actual_pct": None,
        "verdict": "pending",
        "scored_at": None,
    }


class _MockMarketDataFn:
    """market_data_fn 대체용 mock. call_count·calls로 호출 이력을 기록한다."""

    def __init__(self, results=None, sequence=None):
        # results: {market: dict|None} — 매 호출마다 동일 값 반환
        # sequence: [dict|None, ...] — 호출 순서대로 하나씩 반환(그 이후는 마지막 값 반복)
        self._results = results or {}
        self._sequence = sequence
        self.call_count = 0
        self.calls = []

    def __call__(self, market):
        self.call_count += 1
        self.calls.append(market)
        if self._sequence is not None:
            idx = min(self.call_count - 1, len(self._sequence) - 1)
            return self._sequence[idx]
        return self._results.get(market)


class _MockSleepFn:
    """sleep_fn 대체용 mock. 실제로 대기하지 않고 호출만 기록한다."""

    def __init__(self):
        self.call_count = 0
        self.calls = []

    def __call__(self, seconds):
        self.call_count += 1
        self.calls.append(seconds)


class TestNoPending:
    def test_returns_empty_list_immediately(self, monkeypatch, caplog):
        monkeypatch.setattr(scorer.db, "get_pending_forecasts", lambda db_path=None: [])
        update_called = {"flag": False}

        def _update(*args, **kwargs):
            update_called["flag"] = True

        monkeypatch.setattr(scorer.db, "update_forecast_verdict", _update)

        with caplog.at_level("INFO"):
            result = scorer.score_pending_forecasts("2026-07-10T15:40:00")

        assert result == []
        assert update_called["flag"] is False
        assert "채점 대상 없음" in caplog.text


class TestNormalScoring:
    def _run(self, monkeypatch, forecasts, market_data_fn):
        monkeypatch.setattr(scorer.db, "get_pending_forecasts", lambda db_path=None: forecasts)
        updates = []

        def _update(forecast_id, actual_pct, verdict, scored_at, db_path=None):
            updates.append((forecast_id, actual_pct, verdict, scored_at))

        monkeypatch.setattr(scorer.db, "update_forecast_verdict", _update)
        result = scorer.score_pending_forecasts(
            "2026-07-10T15:40:00", market_data_fn=market_data_fn, sleep_fn=_MockSleepFn()
        )
        return result, updates

    def test_up_direction_hit(self, monkeypatch):
        forecasts = [_forecast(1, "KOSPI", "up")]
        md = _MockMarketDataFn(results={"KOSPI": {"close": 3000.0, "change_pct": 1.0, "asof": "d"}})
        result, updates = self._run(monkeypatch, forecasts, md)
        assert result == [{"forecast_id": 1, "market": "KOSPI", "verdict": "hit", "actual_pct": 1.0}]
        assert updates == [(1, 1.0, "hit", "2026-07-10T15:40:00")]

    def test_up_direction_miss(self, monkeypatch):
        forecasts = [_forecast(1, "KOSPI", "up")]
        md = _MockMarketDataFn(results={"KOSPI": {"close": 3000.0, "change_pct": 0.1, "asof": "d"}})
        result, _updates = self._run(monkeypatch, forecasts, md)
        assert result[0]["verdict"] == "miss"

    def test_down_direction_hit(self, monkeypatch):
        forecasts = [_forecast(1, "KOSDAQ", "down")]
        md = _MockMarketDataFn(results={"KOSDAQ": {"close": 800.0, "change_pct": -1.0, "asof": "d"}})
        result, _updates = self._run(monkeypatch, forecasts, md)
        assert result[0]["verdict"] == "hit"

    def test_down_direction_miss(self, monkeypatch):
        forecasts = [_forecast(1, "KOSDAQ", "down")]
        md = _MockMarketDataFn(results={"KOSDAQ": {"close": 800.0, "change_pct": -0.1, "asof": "d"}})
        result, _updates = self._run(monkeypatch, forecasts, md)
        assert result[0]["verdict"] == "miss"

    def test_flat_direction_hit(self, monkeypatch):
        forecasts = [_forecast(1, "KOSPI", "flat")]
        md = _MockMarketDataFn(results={"KOSPI": {"close": 3000.0, "change_pct": 0.0, "asof": "d"}})
        result, _updates = self._run(monkeypatch, forecasts, md)
        assert result[0]["verdict"] == "hit"

    def test_flat_direction_miss(self, monkeypatch):
        forecasts = [_forecast(1, "KOSPI", "flat")]
        md = _MockMarketDataFn(results={"KOSPI": {"close": 3000.0, "change_pct": 1.0, "asof": "d"}})
        result, _updates = self._run(monkeypatch, forecasts, md)
        assert result[0]["verdict"] == "miss"


class TestBoundaryValues:
    def _run_one(self, monkeypatch, direction, change_pct):
        forecasts = [_forecast(1, "KOSPI", direction)]
        md = _MockMarketDataFn(
            results={"KOSPI": {"close": 3000.0, "change_pct": change_pct, "asof": "d"}}
        )
        monkeypatch.setattr(scorer.db, "get_pending_forecasts", lambda db_path=None: forecasts)
        monkeypatch.setattr(scorer.db, "update_forecast_verdict", lambda *a, **k: None)
        result = scorer.score_pending_forecasts("t", market_data_fn=md, sleep_fn=_MockSleepFn())
        return result[0]["verdict"]

    def test_up_plus_0_3_is_hit(self, monkeypatch):
        assert self._run_one(monkeypatch, "up", 0.3) == "hit"

    def test_down_minus_0_3_is_hit(self, monkeypatch):
        assert self._run_one(monkeypatch, "down", -0.3) == "hit"

    def test_up_plus_0_31_is_hit(self, monkeypatch):
        assert self._run_one(monkeypatch, "up", 0.31) == "hit"

    def test_down_minus_0_31_is_hit(self, monkeypatch):
        assert self._run_one(monkeypatch, "down", -0.31) == "hit"

    def test_flat_plus_0_3_is_hit(self, monkeypatch):
        assert self._run_one(monkeypatch, "flat", 0.3) == "hit"

    def test_flat_minus_0_3_is_hit(self, monkeypatch):
        assert self._run_one(monkeypatch, "flat", -0.3) == "hit"

    def test_flat_plus_0_31_is_miss(self, monkeypatch):
        assert self._run_one(monkeypatch, "flat", 0.31) == "miss"

    def test_up_0_29_is_miss(self, monkeypatch):
        assert self._run_one(monkeypatch, "up", 0.29) == "miss"


class TestRetry:
    def test_unavailable_then_success_after_two_retries(self, monkeypatch):
        forecasts = [_forecast(1, "KOSPI", "up")]
        monkeypatch.setattr(scorer.db, "get_pending_forecasts", lambda db_path=None: forecasts)
        updates = []
        monkeypatch.setattr(
            scorer.db,
            "update_forecast_verdict",
            lambda fid, pct, v, t, db_path=None: updates.append((fid, pct, v, t)),
        )
        md = _MockMarketDataFn(
            sequence=[None, None, {"close": 3000.0, "change_pct": 1.0, "asof": "d"}]
        )
        sleep_fn = _MockSleepFn()

        result = scorer.score_pending_forecasts(
            "t", market_data_fn=md, sleep_fn=sleep_fn, max_retries=3, retry_interval_sec=600
        )

        assert md.call_count == 3
        assert sleep_fn.call_count == 2
        assert sleep_fn.calls == [600, 600]
        assert result == [{"forecast_id": 1, "market": "KOSPI", "verdict": "hit", "actual_pct": 1.0}]
        assert updates == [(1, 1.0, "hit", "t")]

    def test_exhausts_retries_stays_pending(self, monkeypatch, caplog):
        forecasts = [_forecast(1, "KOSPI", "up")]
        monkeypatch.setattr(scorer.db, "get_pending_forecasts", lambda db_path=None: forecasts)
        update_called = {"flag": False}

        def _update(*args, **kwargs):
            update_called["flag"] = True

        monkeypatch.setattr(scorer.db, "update_forecast_verdict", _update)
        md = _MockMarketDataFn(sequence=[None, None, None, None])
        sleep_fn = _MockSleepFn()

        with caplog.at_level("WARNING"):
            result = scorer.score_pending_forecasts(
                "t", market_data_fn=md, sleep_fn=sleep_fn, max_retries=3, retry_interval_sec=600
            )

        assert md.call_count == 4
        assert sleep_fn.call_count == 3
        assert update_called["flag"] is False
        assert result == [{"forecast_id": 1, "market": "KOSPI", "verdict": "pending", "actual_pct": None}]
        assert "forecast_id=1" in caplog.text


class TestMarketDataCache:
    def test_same_market_multiple_forecasts_calls_once(self, monkeypatch):
        forecasts = [
            _forecast(1, "KOSPI", "up"),
            _forecast(2, "KOSPI", "down"),
            _forecast(3, "KOSPI", "flat"),
        ]
        monkeypatch.setattr(scorer.db, "get_pending_forecasts", lambda db_path=None: forecasts)
        monkeypatch.setattr(scorer.db, "update_forecast_verdict", lambda *a, **k: None)
        md = _MockMarketDataFn(results={"KOSPI": {"close": 3000.0, "change_pct": 1.0, "asof": "d"}})

        result = scorer.score_pending_forecasts("t", market_data_fn=md, sleep_fn=_MockSleepFn())

        assert md.call_count == 1
        assert len(result) == 3

    def test_different_markets_call_separately(self, monkeypatch):
        forecasts = [
            _forecast(1, "KOSPI", "up"),
            _forecast(2, "KOSDAQ", "up"),
        ]
        monkeypatch.setattr(scorer.db, "get_pending_forecasts", lambda db_path=None: forecasts)
        monkeypatch.setattr(scorer.db, "update_forecast_verdict", lambda *a, **k: None)
        md = _MockMarketDataFn(
            results={
                "KOSPI": {"close": 3000.0, "change_pct": 1.0, "asof": "d"},
                "KOSDAQ": {"close": 800.0, "change_pct": 1.0, "asof": "d"},
            }
        )

        scorer.score_pending_forecasts("t", market_data_fn=md, sleep_fn=_MockSleepFn())

        assert md.call_count == 2


class TestGetHitRate:
    def test_delegates_to_db(self, monkeypatch):
        captured = {}

        def _fake_get_hit_rate(db_path=None):
            captured["db_path"] = db_path
            return 0.75

        monkeypatch.setattr(scorer.db, "get_hit_rate", _fake_get_hit_rate)
        result = scorer.get_hit_rate(db_path="/tmp/x.db")
        assert result == 0.75
        assert captured["db_path"] == "/tmp/x.db"
