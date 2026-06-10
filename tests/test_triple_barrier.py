"""
Triple-Barrier 라벨링 단위 테스트.

테스트 케이스:
  1. TP 배리어 도달 → label=1
  2. SL 배리어 도달 → label=0
  3. 시간 배리어(수익) → label=1
  4. 시간 배리어(손실) → label=0
  5. 같은 날 TP·SL 동시 터치 → SL 우선 → label=0
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.features import triple_barrier_label


def _make_ohlcv(closes, highs=None, lows=None) -> pd.DataFrame:
    """테스트용 최소 OHLCV 데이터프레임 생성."""
    n = len(closes)
    closes = np.array(closes, dtype=float)
    highs  = np.array(highs, dtype=float) if highs is not None else closes
    lows   = np.array(lows,  dtype=float) if lows  is not None else closes
    return pd.DataFrame(
        {
            "Open":   closes,
            "High":   highs,
            "Low":    lows,
            "Close":  closes,
            "Volume": np.ones(n) * 1000,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
    )


TP = 0.07
SL = 0.07
HORIZON = 7


class TestTPBarrier:
    def test_tp_hit_on_day2(self):
        """익절 배리어: 진입 다음 날 High가 TP를 상회 → label=1."""
        entry = 100.0
        highs = [entry, entry * (1 + TP) + 0.1, entry, entry, entry, entry, entry, entry, entry]
        lows  = [entry, entry * 0.99,            entry, entry, entry, entry, entry, entry, entry]
        closes = [entry] * 9
        df = _make_ohlcv(closes, highs=highs, lows=lows)
        labels = triple_barrier_label(df, tp_pct=TP, sl_pct=SL, max_holding_days=HORIZON)
        assert labels.iloc[0] == 1.0, "TP 도달 시 label=1 이어야 함"

    def test_tp_hit_on_last_day(self):
        """TP 배리어가 마지막 허용 거래일에 도달."""
        entry = 100.0
        n = HORIZON + 2  # 1 entry + HORIZON days + 1 buffer
        closes = [entry] * n
        highs = [entry] * n
        lows  = [entry] * n
        highs[HORIZON] = entry * (1 + TP) + 0.1  # 마지막 허용일 TP 도달
        df = _make_ohlcv(closes, highs=highs, lows=lows)
        labels = triple_barrier_label(df, tp_pct=TP, sl_pct=SL, max_holding_days=HORIZON)
        assert labels.iloc[0] == 1.0, "마지막 보유일 TP 도달 시 label=1"


class TestSLBarrier:
    def test_sl_hit_on_day2(self):
        """손절 배리어: 진입 다음 날 Low가 SL을 하회 → label=0."""
        entry = 100.0
        highs  = [entry, entry * 1.01,            entry, entry, entry, entry, entry, entry, entry]
        lows   = [entry, entry * (1 - SL) - 0.1,  entry, entry, entry, entry, entry, entry, entry]
        closes = [entry] * 9
        df = _make_ohlcv(closes, highs=highs, lows=lows)
        labels = triple_barrier_label(df, tp_pct=TP, sl_pct=SL, max_holding_days=HORIZON)
        assert labels.iloc[0] == 0.0, "SL 도달 시 label=0 이어야 함"

    def test_sl_hit_before_tp(self):
        """SL이 TP보다 먼저 도달 → label=0."""
        entry = 100.0
        n = HORIZON + 2
        closes = [entry] * n
        highs = [entry] * n
        lows  = [entry] * n
        # day2: SL 터치
        lows[2]  = entry * (1 - SL) - 0.1
        # day3: TP 터치 (SL이 먼저이므로 무시)
        highs[3] = entry * (1 + TP) + 0.1
        df = _make_ohlcv(closes, highs=highs, lows=lows)
        labels = triple_barrier_label(df, tp_pct=TP, sl_pct=SL, max_holding_days=HORIZON)
        assert labels.iloc[0] == 0.0, "SL이 TP보다 먼저 도달했으므로 label=0"


class TestVerticalBarrier:
    def test_vertical_positive(self):
        """시간 배리어: TP/SL 미도달 + 만료일 종가 > 진입가 → label=1."""
        entry = 100.0
        n = HORIZON + 2
        closes = [entry] * n
        # TP/SL 범위 안에서 소폭 상승 유지, 만료일 종가만 +1%
        closes[HORIZON] = entry * 1.01
        highs = [c * 1.005 for c in closes]   # TP 미도달
        lows  = [c * 0.995 for c in closes]   # SL 미도달
        df = _make_ohlcv(closes, highs=highs, lows=lows)
        labels = triple_barrier_label(df, tp_pct=TP, sl_pct=SL, max_holding_days=HORIZON)
        assert labels.iloc[0] == 1.0, "시간 배리어 + 양수 수익 → label=1"

    def test_vertical_negative(self):
        """시간 배리어: TP/SL 미도달 + 만료일 종가 < 진입가 → label=0."""
        entry = 100.0
        n = HORIZON + 2
        closes = [entry] * n
        closes[HORIZON] = entry * 0.99  # 만료일 -1%
        highs = [c * 1.005 for c in closes]
        lows  = [c * 0.995 for c in closes]
        df = _make_ohlcv(closes, highs=highs, lows=lows)
        labels = triple_barrier_label(df, tp_pct=TP, sl_pct=SL, max_holding_days=HORIZON)
        assert labels.iloc[0] == 0.0, "시간 배리어 + 음수 수익 → label=0"


class TestSimultaneousHit:
    def test_simultaneous_sl_wins(self):
        """같은 날 TP·SL 동시 터치 → SL 우선(보수적 가정) → label=0."""
        entry = 100.0
        n = HORIZON + 2
        closes = [entry] * n
        highs = [entry * 1.001] * n
        lows  = [entry * 0.999] * n
        # day1: 동시 터치
        highs[1] = entry * (1 + TP) + 0.1   # TP 터치
        lows[1]  = entry * (1 - SL) - 0.1   # SL 터치
        df = _make_ohlcv(closes, highs=highs, lows=lows)
        labels = triple_barrier_label(df, tp_pct=TP, sl_pct=SL, max_holding_days=HORIZON)
        assert labels.iloc[0] == 0.0, "동시 터치 시 SL 우선 → label=0"


class TestOutputShape:
    def test_series_length_matches_input(self):
        """출력 Series 길이 = 입력 df 길이."""
        n = 30
        closes = np.linspace(100, 110, n)
        df = _make_ohlcv(closes)
        labels = triple_barrier_label(df, tp_pct=TP, sl_pct=SL, max_holding_days=HORIZON)
        assert len(labels) == n

    def test_labels_binary(self):
        """라벨 값은 0.0 또는 1.0 (NaN 제외)."""
        n = 50
        rng = np.random.default_rng(42)
        closes = 100 + rng.normal(0, 1, n).cumsum()
        df = _make_ohlcv(closes)
        labels = triple_barrier_label(df, tp_pct=TP, sl_pct=SL, max_holding_days=HORIZON)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0}), "라벨은 0.0 또는 1.0만 허용"

    def test_use_intraday_false(self):
        """use_intraday=False: 종가만 사용 → TP/SL 판정이 보수적."""
        entry = 100.0
        n = HORIZON + 2
        closes = [entry] * n
        # High는 TP를 넘지만 Close는 안 넘음
        highs = [entry * (1 + TP) + 0.1] * n
        lows  = [entry * 0.999] * n
        closes[HORIZON] = entry * 0.99  # 만료일 종가 하락
        df = _make_ohlcv(closes, highs=highs, lows=lows)

        label_intraday = triple_barrier_label(
            df, tp_pct=TP, sl_pct=SL, max_holding_days=HORIZON, use_intraday=True
        ).iloc[0]
        label_close_only = triple_barrier_label(
            df, tp_pct=TP, sl_pct=SL, max_holding_days=HORIZON, use_intraday=False
        ).iloc[0]

        # 장중 High 사용 시 TP 도달(1), 종가만 시 SL 미도달·시간 배리어 음수(0)
        assert label_intraday == 1.0
        assert label_close_only == 0.0
