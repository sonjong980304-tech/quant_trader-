"""
config.py - 전략 파라미터 및 환경 설정
모든 숫자 파라미터는 여기서 관리합니다.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# KIS API 설정
# ─────────────────────────────────────────────
KIS_APP_KEY    = os.getenv("KIS_APP_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_APP_SECRET", "")
KIS_ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO", "73018973-01")

# IS_MOCK=True → 모의투자, False → 실투자
IS_MOCK_ENV = os.getenv("KIS_MOCK", "true").lower()
IS_MOCK = IS_MOCK_ENV == "true"

# 모의/실투자에 따라 엔드포인트 자동 전환
if IS_MOCK:
    KIS_BASE_URL = "https://openapivts.koreainvestment.com:29443"
else:
    KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"

# ─────────────────────────────────────────────
# 외부 API 설정
# ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY", "")

# ─────────────────────────────────────────────
# 매매 대상 종목 (yfinance 티커 : 종목명)
# 한국 주식은 yfinance에서 ".KS" 접미사 사용
# ─────────────────────────────────────────────
STOCKS = {
    "010140.KS": "삼성중공업",
    "005290.KS": "동진세미켐",
    "047040.KS": "대우건설",
    "432720.KQ": "퀄리타스반도체",
    "028050.KS": "삼성E&A",
    "120110.KS": "코오롱인더",
    "117700.KS": "KODEX 증권 ETF",
}

# ─────────────────────────────────────────────
# 전략 A (기본): 단기 골든크로스 + RSI 모멘텀
# ─────────────────────────────────────────────
STRATEGY_A = {
    "name": "전략A_MA5_20",
    "short_window": 5,    # 단기 이평선
    "long_window": 20,    # 장기 이평선
    "rsi_period": 14,     # RSI 계산 기간
    "rsi_buy_threshold": 55,   # 매수 RSI 최소값
    "rsi_overbought": 75,      # 과매수 진입 기준
    "rsi_overbought_exit": 70, # 과매수 청산 기준
    "partial_sell_ratio": 0.5, # 분할매도 비율 50%
}

# ─────────────────────────────────────────────
# 전략 B (비교): 중기 골든크로스 + RSI 모멘텀
# ─────────────────────────────────────────────
STRATEGY_B = {
    "name": "전략B_MA20_60",
    "short_window": 20,
    "long_window": 60,
    "rsi_period": 14,
    "rsi_buy_threshold": 50,
    "rsi_overbought": 75,
    "rsi_overbought_exit": 70,
    "partial_sell_ratio": 0.5,
}

# 백테스트에 사용할 기본 전략
ACTIVE_STRATEGY = STRATEGY_A

# ─────────────────────────────────────────────
# 백테스트 설정
# ─────────────────────────────────────────────
BACKTEST_PERIOD_YEARS = 3      # 최근 몇 년치 데이터
BACKTEST_INIT_CASH    = 10_000_000  # 초기 자금 1천만원

# ─────────────────────────────────────────────
# 리스크 관리
# ─────────────────────────────────────────────
MAX_CONSECUTIVE_BUY = 3   # 연속 매수 신호 경고 임계값
ORDER_AMOUNT        = 200_000  # 1회 주문 금액 (원)

# ─────────────────────────────────────────────
# 로그 설정
# ─────────────────────────────────────────────
LOG_FILE = "logs/trader.log"
