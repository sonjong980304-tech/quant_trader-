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
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
NAVER_CLIENT_ID     = os.getenv("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "")

# 뉴스 브리핑 LLM 모델 (news_briefing 패키지 전용) — 실제 API 접근 가능 확인된 티어로 비용 최적화 분리
# CHEAP(gpt-5.4-nano): 기사 선별(selector)·근거 판정(verifier.check_grounding) — 고빈도·단순 분류 작업
# PREMIUM(gpt-5.4): closed-book 작성·재생성(writer) — 품질이 중요한 실제 문장 생성 작업
NEWS_LLM_CHEAP      = os.getenv("NEWS_LLM_CHEAP", "gpt-5.4-nano")
NEWS_LLM_PREMIUM    = os.getenv("NEWS_LLM_PREMIUM", "gpt-5.4")

# ─────────────────────────────────────────────
# 매매 대상 종목 (stocks.py에서 로드 — GitHub 비공개)
# ─────────────────────────────────────────────
try:
    from stocks import STOCKS, US_STOCKS
except ImportError:
    STOCKS = {}
    US_STOCKS = {}

# ─────────────────────────────────────────────
# 이동평균선 파라미터
# ─────────────────────────────────────────────
MA_SHORT   = 5    # 단기 이동평균선 (5일)
MA_LONG    = 20   # 장기 이동평균선 (20일)
RSI_PERIOD = 14   # RSI 계산 기간 (보조 지표)

# ─────────────────────────────────────────────
# 캔들 판단 파라미터
# ─────────────────────────────────────────────
DOJI_THRESHOLD          = 0.1   # 도지: 몸통/전체범위 비율 기준
LARGE_CANDLE_MULTIPLIER = 1.5   # 장대봉: 직전 5일 평균 캔들 대비 배수

# ─────────────────────────────────────────────
# 거래량 파라미터
# ─────────────────────────────────────────────
VOLUME_INCREASE_RATIO      = 1.5   # 거래량 증가 기준 (평균 대비 배수)
VOLUME_SURGE_RATIO         = 2.0   # 거래량 급증 기준 (평균 대비 배수, 일봉)
VOLUME_SURGE_MINUTE_RATIO  = 5.0   # 분봉 거래량 급증 기준 (직전 평균 대비 배수)
VOLUME_LOOKBACK_DAYS       = 50    # 평균 거래량 계산 기간
PULLBACK_DAYS              = 3     # 눌림목 판단 최소 연속 일수 (매수 2원칙)

# ─────────────────────────────────────────────
# 백테스트 설정
# ─────────────────────────────────────────────
BACKTEST_PERIOD_YEARS = 3           # 최근 몇 년치 데이터
BACKTEST_INIT_CASH    = 10_000_000  # 초기 자금 1천만원

# ─────────────────────────────────────────────
# 리스크 관리
# ─────────────────────────────────────────────
MAX_CONSECUTIVE_BUY = 3        # 연속 매수 신호 경고 임계값
ORDER_AMOUNT        = 200_000  # 1회 주문 금액 (원, 백테스트 등 참고용)
ORDER_RATIO         = 0.40    # 매수 신호 시 가용 현금 대비 주문 비율 (40%)

# ─────────────────────────────────────────────
# 손절 설정
# ─────────────────────────────────────────────
STOP_LOSS_HARD        = -0.06   # 매수가 대비 -6% : 절대 하방 손절
STOP_LOSS_WARN        = -0.03   # 매수가 대비 -3% : 경고 알림
TRAILING_STOP_RATIO   = -0.08   # 고점 대비 -8%   : 트레일링 스탑
MA_STOP_ENABLED       = True    # 5일선 종가 이탈 손절 활성화 여부

# ─────────────────────────────────────────────
# 익절 설정
# ─────────────────────────────────────────────
TAKE_PROFIT_HALF      = 0.08    # +8%  : 보유량 50% 1차 익절
TAKE_PROFIT_FULL      = 0.15    # +15% : 전량 2차 익절

# ─────────────────────────────────────────────
# 자금 배분
# ─────────────────────────────────────────────
MAX_POSITION_RATIO    = 0.15    # 1종목 기본 비중 15%
MAX_POSITION_LIMIT    = 0.20    # 1종목 최대 비중 20%

# ─────────────────────────────────────────────
# 이동평균선 추가 설정
# ─────────────────────────────────────────────
MA20_RISING_LOOKBACK  = 3       # MA20 우상향 판단 기간 (일)

# ─────────────────────────────────────────────
# 장 시간 설정
# ─────────────────────────────────────────────
MARKET_OPEN_HOUR      = 9
MARKET_OPEN_MIN       = 0
MARKET_CLOSE_HOUR     = 15
MARKET_CLOSE_MIN      = 30
MARKET_TOTAL_MINUTES  = 390     # 9:00 ~ 15:30 = 390분

# ─────────────────────────────────────────────
# 로그 설정
# ─────────────────────────────────────────────
LOG_FILE = "logs/trader.log"

# ─────────────────────────────────────────────
# 페이퍼 트레이딩 기준값 (슬롯 분리 10+10, 2026-06-19 채택)
# ─────────────────────────────────────────────
# 합산 백테스트: reversion 10슬롯 + trend 10슬롯 분리 운용
PAPER_BACKTEST_EV_KR = None       # 구시대 WF OOF 기준값 — 현재 전략과 무관하여 비활성화
PAPER_BACKTEST_EV_US = None      # US 미운용

# reversion 에이전트 전용 파라미터
TP_PCT               = 0.15      # reversion 익절 +15%
SL_PCT               = 0.08      # reversion 손절 -8%
SPLIT_GUARD_PCT      = 0.30      # 진입가 대비 ±30% 이상 급변 시 액면분할/데이터 이상으로 간주 — 자동매매 보류
EOD_SLIPPAGE_PCT     = 0.0005    # 0.05% 슬리피지
EOD_HORIZON          = 10        # reversion 보유기간 (거래일)
# trend 에이전트: TP 없음, trailing stop 2.0×ATR + MA20 이탈 청산

# 슬롯 설정
REV_SLOTS            = 10        # reversion 전용 슬롯
TR_SLOTS             = 10        # trend 전용 슬롯
MAX_TOTAL_SLOTS      = 20        # 총 최대 동시 보유 종목 수

# ─────────────────────────────────────────────
# 실거래 안전 게이트
# ─────────────────────────────────────────────
# LIVE_TRADING=False: 페이퍼 트레이딩 모드 (2주 검증 후 True 전환 예정)
# 전환 조건: EV>0, CI 하단>0, 슬리피지<0.5%
LIVE_TRADING = False

# ─────────────────────────────────────────────
# ML / 퀀트 설정
# ─────────────────────────────────────────────
ML_HORIZON          = 7      # 예측 기간 (일)
ML_THRESHOLD        = 0.03   # 성공 기준 수익률 (3% 이상 = 성공)
ML_MIN_WIN_PROB     = 0.60   # 신호 발송 최소 승률 (0.58→0.60, 2026-07-01 상향)
ML_MIN_RISK_REWARD  = 1.5    # 신호 발송 최소 손익비

# ─────────────────────────────────────────────
# ML 재학습 스케줄 (분기말 다음달 1일, 연 4회)
# ─────────────────────────────────────────────
# 1/4/7/10월 1일. 해당일이 휴장이면 다음 영업일로 자동 보정.
# 추론(일일 신호 스캔)은 기존대로 매일 15:31 유지.
RETRAIN_SCHEDULE = ['01-01', '04-01', '07-01', '10-01']
