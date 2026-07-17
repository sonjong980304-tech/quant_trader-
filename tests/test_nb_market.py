"""
tests/test_nb_market.py — news_briefing.market_data / news_briefing.positions 검증

yfinance / FinanceDataReader / KIS(dashboard.kis_live) 는 전부 sys.modules mock 처리한다.
FinanceDataReader는 테스트 venv(3.11)에 설치돼 있지 않으므로, market_data.py는
모듈 최상단이 아닌 함수 내부 지연 임포트를 강제한다 — 이 계약을 별도 테스트로 명시 검증한다.
"""

import json
import os
import sys
import types

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────
# 지연 임포트 계약 검증
# ─────────────────────────────────────────────

class TestLazyImport:
    def test_import_succeeds_without_financedatareader(self, monkeypatch):
        """FinanceDataReader가 없는 환경(테스트 venv 기본 상태)에서도
        import news_briefing.market_data 자체는 성공해야 한다."""
        monkeypatch.delitem(sys.modules, "FinanceDataReader", raising=False)
        monkeypatch.delitem(sys.modules, "news_briefing.market_data", raising=False)
        import news_briefing.market_data  # noqa: F401 — import 성공 자체가 검증 대상


# ─────────────────────────────────────────────
# get_us_snapshot
# ─────────────────────────────────────────────

def _mock_yf_module(history_map: dict) -> types.ModuleType:
    """yfinance 모듈 mock. history_map: {ticker: DataFrame(Close 컬럼)}"""
    mod = types.ModuleType("yfinance")

    class _MockTicker:
        def __init__(self, ticker):
            self._ticker = ticker

        def history(self, period="5d"):
            return history_map.get(self._ticker, pd.DataFrame())

    mod.Ticker = _MockTicker
    return mod


class TestGetUsSnapshot:
    def test_change_pct_calculation(self, monkeypatch):
        import news_briefing.market_data as md

        idx = pd.to_datetime(["2026-07-08", "2026-07-09"])
        history_map = {
            "^IXIC": pd.DataFrame({"Close": [20000.0, 20200.0]}, index=idx),
            "^GSPC": pd.DataFrame({"Close": [5000.0, 4950.0]}, index=idx),
            "^DJI": pd.DataFrame({"Close": [40000.0, 40400.0]}, index=idx),
            "^SOX": pd.DataFrame({"Close": [5000.0, 4900.0]}, index=idx),
        }
        monkeypatch.setitem(sys.modules, "yfinance", _mock_yf_module(history_map))

        snap = md.get_us_snapshot()

        assert snap["nasdaq"]["close"] == 20200.0
        assert snap["nasdaq"]["change_pct"] == pytest.approx(1.0)
        assert snap["sp500"]["close"] == 4950.0
        assert snap["sp500"]["change_pct"] == pytest.approx(-1.0)
        assert snap["dow"]["close"] == 40400.0
        assert snap["dow"]["change_pct"] == pytest.approx(1.0)
        assert snap["sox"]["close"] == 4900.0
        assert snap["sox"]["change_pct"] == pytest.approx(-2.0)
        assert snap["asof"] == "2026-07-09"

    def test_change_pct_and_close_rounded_to_two_decimals(self, monkeypatch):
        """LLM 프롬프트·발송 메시지에 0.28511108...% 같은 과도한 정밀도가 노출되지
        않도록 close·change_pct는 소수점 2자리로 반올림된다."""
        import news_briefing.market_data as md

        idx = pd.to_datetime(["2026-07-08", "2026-07-09"])
        history_map = {
            "^IXIC": pd.DataFrame({"Close": [20000.0, 20057.022]}, index=idx),
            "^GSPC": pd.DataFrame({"Close": [5000.0, 5000.0]}, index=idx),
            "^DJI": pd.DataFrame({"Close": [40000.0, 40000.0]}, index=idx),
            "^SOX": pd.DataFrame({"Close": [5000.0, 5000.0]}, index=idx),
        }
        monkeypatch.setitem(sys.modules, "yfinance", _mock_yf_module(history_map))

        snap = md.get_us_snapshot()

        assert snap["nasdaq"]["change_pct"] == 0.29  # (57.022/20000*100)=0.28511... -> 0.29
        assert snap["nasdaq"]["close"] == 20057.02

    def test_partial_ticker_failure_keeps_others(self, monkeypatch):
        """일부 지수(예: SOX) 조회가 실패해도 나머지 지수는 정상 반환된다."""
        import news_briefing.market_data as md

        idx = pd.to_datetime(["2026-07-08", "2026-07-09"])
        history_map = {
            "^IXIC": pd.DataFrame({"Close": [20000.0, 20200.0]}, index=idx),
            "^GSPC": pd.DataFrame({"Close": [5000.0, 4950.0]}, index=idx),
            "^DJI": pd.DataFrame({"Close": [40000.0, 40400.0]}, index=idx),
            # ^SOX 누락 — 해당 종목만 조회 실패 시나리오
        }
        monkeypatch.setitem(sys.modules, "yfinance", _mock_yf_module(history_map))

        snap = md.get_us_snapshot()

        assert "sox" not in snap
        assert snap["nasdaq"]["close"] == 20200.0
        assert snap["dow"]["close"] == 40400.0

    def test_returns_empty_dict_on_total_failure(self, monkeypatch):
        import news_briefing.market_data as md

        monkeypatch.setitem(sys.modules, "yfinance", _mock_yf_module({}))
        snap = md.get_us_snapshot()
        assert snap == {}


# ─────────────────────────────────────────────
# get_kr_index_change / get_kr_index_snapshot
# ─────────────────────────────────────────────

def _mock_fdr_module(data_map: dict) -> types.ModuleType:
    """FinanceDataReader 모듈 mock. data_map: {code: DataFrame(Close 컬럼)}"""
    mod = types.ModuleType("FinanceDataReader")

    def _datareader(code, *args, **kwargs):
        if code not in data_map:
            raise ValueError(f"unknown code: {code}")
        return data_map[code]

    mod.DataReader = _datareader
    return mod


class TestGetKrIndexChange:
    def test_kospi_change_pct(self, monkeypatch):
        import news_briefing.market_data as md

        idx = pd.to_datetime(["2026-07-08", "2026-07-09"])
        df = pd.DataFrame({"Close": [3000.0, 3060.0]}, index=idx)
        monkeypatch.setitem(sys.modules, "FinanceDataReader", _mock_fdr_module({"KS11": df}))

        result = md.get_kr_index_change("KOSPI")
        assert result["close"] == 3060.0
        assert result["change_pct"] == pytest.approx(2.0)
        assert result["asof"] == "2026-07-09"

    def test_kosdaq_change_pct(self, monkeypatch):
        import news_briefing.market_data as md

        idx = pd.to_datetime(["2026-07-08", "2026-07-09"])
        df = pd.DataFrame({"Close": [800.0, 792.0]}, index=idx)
        monkeypatch.setitem(sys.modules, "FinanceDataReader", _mock_fdr_module({"KQ11": df}))

        result = md.get_kr_index_change("KOSDAQ")
        assert result["close"] == 792.0
        assert result["change_pct"] == pytest.approx(-1.0)

    def test_returns_none_when_less_than_two_rows(self, monkeypatch):
        import news_briefing.market_data as md

        idx = pd.to_datetime(["2026-07-09"])
        df = pd.DataFrame({"Close": [3000.0]}, index=idx)
        monkeypatch.setitem(sys.modules, "FinanceDataReader", _mock_fdr_module({"KS11": df}))

        assert md.get_kr_index_change("KOSPI") is None

    def test_returns_none_on_exception(self, monkeypatch):
        import news_briefing.market_data as md

        # data_map에 KS11이 없어 mock DataReader가 예외를 던지는 상황(네트워크 실패 등 대체)
        monkeypatch.setitem(sys.modules, "FinanceDataReader", _mock_fdr_module({}))
        assert md.get_kr_index_change("KOSPI") is None

    def test_invalid_market_returns_none(self):
        import news_briefing.market_data as md

        assert md.get_kr_index_change("NYSE") is None


class TestGetKrIndexSnapshot:
    def test_bundles_kospi_and_kosdaq(self, monkeypatch):
        import news_briefing.market_data as md

        idx = pd.to_datetime(["2026-07-08", "2026-07-09"])
        kospi_df = pd.DataFrame({"Close": [3000.0, 3030.0]}, index=idx)
        kosdaq_df = pd.DataFrame({"Close": [800.0, 796.0]}, index=idx)
        monkeypatch.setitem(
            sys.modules,
            "FinanceDataReader",
            _mock_fdr_module({"KS11": kospi_df, "KQ11": kosdaq_df}),
        )

        snap = md.get_kr_index_snapshot()
        assert snap["kospi"]["close"] == 3030.0
        assert snap["kosdaq"]["close"] == 796.0

    def test_missing_data_yields_none_entry(self, monkeypatch):
        import news_briefing.market_data as md

        idx = pd.to_datetime(["2026-07-08", "2026-07-09"])
        kospi_df = pd.DataFrame({"Close": [3000.0, 3030.0]}, index=idx)
        monkeypatch.setitem(
            sys.modules, "FinanceDataReader", _mock_fdr_module({"KS11": kospi_df})
        )

        snap = md.get_kr_index_snapshot()
        assert snap["kospi"] is not None
        assert snap["kosdaq"] is None


# ─────────────────────────────────────────────
# get_holdings
# ─────────────────────────────────────────────

class TestGetHoldings:
    def _write_paper_positions(self, tmp_path, monkeypatch, data):
        import news_briefing.positions as pos

        path = tmp_path / "paper_positions.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(pos, "_PAPER_POSITIONS_PATH", str(path))
        return pos

    def _mock_kis_live(self, monkeypatch, balance, trader_available=True):
        mod = types.ModuleType("kis_live")

        class _MockTrader:
            def get_balance(self):
                return balance

        def _get_trader():
            return _MockTrader() if trader_available else None

        mod._get_trader = _get_trader
        monkeypatch.setitem(sys.modules, "kis_live", mod)
        return mod

    def test_union_marks_both_when_overlap(self, tmp_path, monkeypatch):
        pos = self._write_paper_positions(tmp_path, monkeypatch, {
            "a": {"ticker": "005930.KS", "name": "삼성전자"},
            "b": {"ticker": "009150.KS", "name": "삼성전기"},
        })
        self._mock_kis_live(monkeypatch, balance=[
            {"stock_code": "005930", "name": "삼성전자", "qty": 10},
        ])

        holdings = pos.get_holdings()
        by_ticker = {h["ticker"]: h for h in holdings}

        assert by_ticker["005930.KS"]["source"] == "both"
        assert by_ticker["009150.KS"]["source"] == "paper"

    def test_kis_only_holding_marked_kis(self, tmp_path, monkeypatch):
        pos = self._write_paper_positions(tmp_path, monkeypatch, {})
        self._mock_kis_live(monkeypatch, balance=[
            {"stock_code": "005930", "name": "삼성전자", "qty": 10},
        ])

        holdings = pos.get_holdings()
        assert len(holdings) == 1
        assert holdings[0]["ticker"] == "005930"
        assert holdings[0]["source"] == "kis"
        assert holdings[0]["name"] == "삼성전자"

    def test_kis_failure_falls_back_to_paper_only(self, tmp_path, monkeypatch):
        pos = self._write_paper_positions(tmp_path, monkeypatch, {
            "a": {"ticker": "005930.KS", "name": "삼성전자"},
        })

        mod = types.ModuleType("kis_live")

        def _get_trader():
            raise ConnectionError("network down")

        mod._get_trader = _get_trader
        monkeypatch.setitem(sys.modules, "kis_live", mod)

        holdings = pos.get_holdings()  # 예외 전파 없이 페이퍼만으로 진행
        assert len(holdings) == 1
        assert holdings[0]["ticker"] == "005930.KS"
        assert holdings[0]["source"] == "paper"

    def test_kis_trader_none_falls_back_to_paper_only(self, tmp_path, monkeypatch):
        pos = self._write_paper_positions(tmp_path, monkeypatch, {
            "a": {"ticker": "005930.KS", "name": "삼성전자"},
        })
        self._mock_kis_live(monkeypatch, balance=[], trader_available=False)

        holdings = pos.get_holdings()
        assert len(holdings) == 1
        assert holdings[0]["source"] == "paper"

    def test_missing_paper_file_returns_kis_only(self, tmp_path, monkeypatch):
        import news_briefing.positions as pos

        monkeypatch.setattr(pos, "_PAPER_POSITIONS_PATH", str(tmp_path / "does_not_exist.json"))
        self._mock_kis_live(monkeypatch, balance=[
            {"stock_code": "005930", "name": "삼성전자", "qty": 10},
        ])

        holdings = pos.get_holdings()
        assert len(holdings) == 1
        assert holdings[0]["source"] == "kis"
