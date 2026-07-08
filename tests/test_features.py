"""
tests/test_features.py — reversion 피처셋 회귀 가드

FEATURE_COLS_REVERSION은 4년(2023~2026) walk-forward 백테스트로 검증된 축소안.
검증 근거:
  - Permutation importance 분석: 기존 12개 중 kospi_relative_5d/candle_body/rsi/low52_pct
    4개만 검증 AUC에 실질 기여, 나머지(atr_pct 등)는 오히려 검증 성능을 깎는 노이즈였음.
  - 축소(4개) vs 기존(12개) 동일 캐시/유니버스 백테스트: 2023~2026 네 개 연도 전부 개선,
    특히 2024년은 손실(-3.5%)→흑자(+5.2%) 전환, 6~7월 급락구간 -134.1%p→-5.9%p.
이 테스트는 향후 무심코 피처를 되돌리거나 추가하는 것을 막는 회귀 가드다.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.features import FEATURE_COLS_REVERSION


class TestFeatureColsReversion:
    def test_reduced_to_validated_four_features(self):
        assert FEATURE_COLS_REVERSION == [
            "kospi_relative_5d", "candle_body", "rsi", "low52_pct",
        ]

    def test_noise_features_excluded(self):
        # permutation importance 음수(검증 성능을 깎음) — 재도입 금지
        noise = {"atr_pct", "bb_pct_20", "ret_5d", "ret_3d", "bb_std_20"}
        assert noise.isdisjoint(set(FEATURE_COLS_REVERSION))
