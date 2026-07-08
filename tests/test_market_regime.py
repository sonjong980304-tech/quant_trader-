"""
tests/test_market_regime.py — 레짐 판정 순수 함수 테스트

regime_allows_trend(close, ma200, dd20):
  trend 에이전트 진입 게이트. 순수 계산으로 판정한다.
  조건: close > ma200 AND dd20 > -0.05
    - close > ma200 : 지수가 200일선 위 (상승 추세 포착 — 상승장 수익 유지)
    - dd20   > -0.05 : 20일 고점 대비 낙폭 -5% 이내 (급락 국면 신규 진입 차단)

  out-of-sample 검증(2024-12~2026-02)과 in-sample(2026-03~07) 두 기간 모두
  MA200 단독 필터를 상회한 유일한 견고 후보.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from market_regime import regime_allows_trend


class TestRegimeAllowsTrend:
    def test_allows_uptrend_small_drawdown(self):
        # MA200 위 + 낙폭 -2% (정상 상승 국면) → 진입 허용
        assert regime_allows_trend(close=100.0, ma200=90.0, dd20=-0.02) is True

    def test_blocks_below_ma200(self):
        # 종가가 MA200 아래 → 차단 (하락 추세)
        assert regime_allows_trend(close=85.0, ma200=90.0, dd20=-0.02) is False

    def test_blocks_large_drawdown_even_above_ma200(self):
        # MA200 위지만 20일 낙폭 -8% (급락 국면) → 차단 (핵심: 과열 후 조정 회피)
        assert regime_allows_trend(close=100.0, ma200=90.0, dd20=-0.08) is False

    def test_blocks_at_drawdown_boundary(self):
        # 낙폭 정확히 -5%는 '> -0.05'를 만족하지 않으므로 차단 (경계값)
        assert regime_allows_trend(close=100.0, ma200=90.0, dd20=-0.05) is False

    def test_allows_at_zero_drawdown(self):
        # 신고가 부근(낙폭 0) + MA200 위 → 허용
        assert regime_allows_trend(close=100.0, ma200=90.0, dd20=0.0) is True
