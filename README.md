# 퀀트 자동매매 시스템

> **투자 책임 고지**: 이 프로그램은 교육 및 연구 목적으로 제작되었습니다.  
> 실제 투자 손익에 대한 책임은 전적으로 사용자 본인에게 있습니다.  
> 과거 성과가 미래 수익을 보장하지 않습니다.

---

## 포트폴리오 구조

### 70% — 안전자산 (몬테카를로 시뮬레이션 최적화)

| 자산 | 비중 | 설명 |
|------|------|------|
| QQQ | 22.3% | 나스닥 100 ETF |
| 삼성전자 (005930.KS) | 27.3% | 국내 대형주 |
| TLT | 0.2% | 미국 장기채 ETF |
| ACE KRX금현물 (411060.KS) | 50.3% | 금 ETF |

- 최신 5개년 데이터 기반 몬테카를로 시뮬레이션 10만 번 → **최대 샤프비율** 비중 도출
- 매월 1일 08:30 자동 리밸런싱 (LLM이 주식 수량 결정)

### 30% — 급등주 (ML 전략)

- **XGBoost** + TimeSeriesSplit(5-fold)로 7일 후 수익률 예측
- **5분봉 장중 실시간** 신호 감지 → 다음 봉 시가에 즉시 진입
- 진입 조건: 기술적 트리거 감지 AND 승률 ≥ 55% AND 손익비 ≥ 1.5
- 포지션 사이징: **켈리 공식** (최대 25% 한도)
- 매일 07:30 **5년치 데이터로 자동 재학습** (최신 시장 반영)

---

## ML 전략 상세

### 유니버스 스크리닝 (1단계)

| 시장 | 방법 | 조건 |
|------|------|------|
| 한국 | pykrx 전체 종목 | 등락률 > 0% + 거래량 비율 상위 100개 |
| 미국 | S&P 500 전체 (503종목) | 등락률 > 0% + 거래량 비율 × 1.5 이상 상위 50개 |

- 거래량은 **한국(KST 09:00~15:30) / 미국(ET 09:30~16:00)** 장 시간 기준으로 하루 예상 거래량 환산
- 구조적 하락 종목은 **블랙리스트**(`BLACKLIST`)로 영구 제외

### 기술적 트리거 (2단계)

| 신호 | 조건 |
|------|------|
| 거래량폭발 | 예상 하루 거래량 > 20일 평균 × 2.0배 + 양봉 |
| BB하단반등 | 종가가 볼린저밴드 하단 이탈 후 재진입 |
| RSI과매도탈출 | RSI 30 이하에서 30 돌파 |
| 이격도저점 | EMA20 대비 -5% 이하 이격 |
| BB스퀴즈돌파 | 밴드 수축(60일 최저) 후 상단 돌파 |

> 분봉 누적으로 오늘 일봉 바를 실시간 합성하여 트리거 감지

### ML 예측 (3단계)

**XGBoost 피처 (15종)**

`change_rate`, `volume_change`, `rsi`, `ema_deviation_20`, `bb_width_20`, `bb_pct_20`, `bb_std_20`, `volume_ratio`, `candle_body`, `candle_upper_wick`, `candle_lower_wick`, `ret_3d`, `ret_5d`, `ret_10d`, `volatility_10d`

**학습 데이터**: 5년치 일봉 / 매일 07:30 자동 재학습

### 하프켈리 포지션 사이징

```
풀켈리: f* = (p × b - q) / b
하프켈리: f = f* × 0.5  ← 실제 적용값

p = ML 예측 승률
b = 평균 수익 / 평균 손실 (손익비)
q = 1 - p
```

풀켈리는 입력값(승률·손익비) 추정 오차에 민감하므로 하프켈리로 완충합니다.

---

## 자동화 스케줄

| 시간 | 동작 |
|------|------|
| 07:30 (평일) | ML 모델 전종목 재학습 (5년치 데이터) |
| 08:00 (평일) | 모닝 브리핑 (보유종목 뉴스 + 시황) |
| 08:30 (매월 1일) | 안전자산 몬테카를로 리밸런싱 |
| 5분 간격 (한국장 09:00~15:30) | 분봉 기반 급등주 ML 신호 스캔 |
| 5분 간격 (미국장 ET 09:30~16:00) | S&P 500 분봉 기반 급등주 ML 신호 스캔 |
| 15:00 (평일) | 일일 기술적 분석 리포트 |

---

## 봇 활성화 게이트

기존 보유 종목이 있을 경우, 해당 종목을 **직접 모두 매도**해야 자동매매가 시작됩니다.

```
state.json: {"bot_active": false, "legacy_tickers": ["XXXX"], ...}
  ↓ 직접 매도 완료
state.json: {"bot_active": true, ...}  →  자동매매 시작
```

텔레그램 LLM 기능(질문, 브리핑 등)은 봇 활성화 여부와 무관하게 항상 작동합니다.

---

## 미국주식 — 통합증거금서비스

QQQ, TLT 등 미국주식은 **KIS 통합증거금서비스**를 통해 거래합니다.  
원화 잔고에서 자동으로 환전되어 USD 결제가 처리되므로 별도 환전 불필요합니다.

---

## 매매 이력 관리

모든 자동매매 거래는 `trade_history.csv`에 자동 기록됩니다.

| 컬럼 | 설명 |
|------|------|
| trade_id | 거래 고유 ID |
| ticker / name | 종목코드 / 종목명 |
| entry_date / entry_price | 매수일 / 매수가 |
| exit_date / exit_price | 매도일 / 매도가 |
| qty | 수량 |
| pnl_amount / pnl_pct | 손익(원) / 손익률(%) |
| win | 성공(1) / 실패(0) |
| strategy | 사용 전략 |

거래 업데이트마다 CSV 파일을 텔레그램으로 자동 전송합니다.

---

## GPT AI 어시스턴트 (LangChain)

`ConversationBufferWindowMemory(k=10)`로 대화 맥락을 유지합니다.

| 툴 | 용도 |
|----|------|
| `get_naver_finance` | 한국 주식 재무지표 (PER, PBR, EPS 등) |
| `get_yahoo_finance` | 미국 주식 재무지표 |
| `get_naver_news` | 네이버 최신 뉴스 검색 |
| `get_stock_signal` | 기술적 지표 + 매수/매도 신호 분석 |
| `get_historical_price` | 특정 날짜 종목 종가 조회 |
| `get_account_balance` | 국내 + 미국주식 잔고 조회 |
| `get_portfolio_status` | 안전자산 포트폴리오 현황 + 리밸런싱 필요 여부 |
| `set_conditional_order` | 조건부 주문 등록 |
| `list_conditional_orders` | 조건부 주문 목록 조회 |
| `cancel_conditional_order` | 조건부 주문 취소 |

툴을 사용한 답변에는 **Context Recall 점수**(0.0~1.0)가 자동 표시됩니다 (gpt-4.5-mini 평가).

---

## 텔레그램 명령어

| 명령어 | 설명 |
|--------|------|
| 자유 텍스트 | LangChain AI 어시스턴트 자동 답변 |
| `/ask <질문>` | GPT 명시적 질문 |
| `/reset` | 대화 기록 초기화 |
| `/status` | 전 종목 신호 조회 |
| `/balance` | 국내 + 미국주식 잔고 조회 |
| `/portfolio` | 안전자산 70% 포트폴리오 현황 |
| `/scanstocks` | 급등주 ML 신호 수동 스캔 |
| `/buysignal_TICKER` | 스캔 신호 매수 확정 |
| `/skipsignal` | 대기 신호 패스 |
| `/trainmodel` | ML 모델 전체 재학습 |
| `/tradestats` | 매매 이력 통계 + CSV 전송 |
| `/backtest` | 45일 분봉 ML 백테스트 |
| `/stocks` | 관심종목 목록 |
| `/addstock 코드 이름` | 종목 추가 |
| `/removestock 코드` | 종목 삭제 |
| `/buy 코드 수량` | 수동 매수 |
| `/sell 코드 수량` | 수동 매도 |
| `/sellall 코드` | 전량 매도 |

---

## 전체 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│                  macOS launchd — 3개 데몬 상시 실행                    │
├──────────────┬───────────────────────────┬───────────────────────────┤
│  runner.py   │  telegram_bot.py          │  dashboard.py             │
│  (스케줄러)   │  (사용자 인터페이스)        │  (Streamlit 모니터링)       │
└──────┬───────┴─────────────┬─────────────┴───────────────────────────┘
       │                     │
       │    ┌────────────────▼──────────────────────────────────────┐
       │    │         langchain_agent.py  (LangChain + GPT-4.1)     │
       │    │   ConversationBufferWindowMemory(k=10) 유저별 유지      │
       │    │   10종 툴 자동 호출 + Context Recall 평가              │
       │    └───────────────────────────────────────────────────────┘
       │
       ├── [07:30] ml/trainer.py — 전종목 XGBoost 재학습 (5년치)
       │
       ├── [08:00] morning_briefer.py — 보유종목 뉴스 + 시황
       │
       ├── [08:30/매월1일] portfolio/rebalancer.py
       │     몬테카를로 10만번 → 최대 샤프비율 비중
       │     → GPT-4.1이 수량 결정 → KIS 주문
       │
       ├── [5분/한국장·미국장] scan_growth_signals()
       │     signals/krx_universe.py → KRX 전체 1차 스크리닝 (100개)
       │     signals/us_universe.py  → S&P 500 전체 1차 스크리닝 (50개)
       │       ↓ yfinance 5분봉 + 오늘 바 합성
       │     signals/scanner.py → 기술적 트리거 5종 감지
       │       ↓ XGBoost 예측 (승률≥55% AND 손익비≥1.5)
       │     → 텔레그램 알림 → 사용자 /buysignal 확인 → 켈리 공식 매수
       │
       └── [15:00] send_daily_summary() — 일일 기술적 분석 리포트

공통 레이어:
  yfinance (일봉·5분봉)  ·  pykrx (KRX 유니버스)  ·  KIS API (주문·잔고)
  ml/models/*.pkl  ·  trade_history.csv  ·  state.json
```

---

## 프로젝트 구조

```
quant_trader/
├── config.py               # 전략 파라미터 / API 설정
├── stocks.py               # 관심종목 (STOCKS, US_STOCKS)
├── runner.py               # 스케줄러 (07:30 재학습·스캔·리밸런싱)
├── telegram_bot.py         # 텔레그램 봇
├── langchain_agent.py      # LangChain AI 어시스턴트
├── trader.py               # KIS API (국내 + 미국주식)
├── trade_logger.py         # 매매 이력 CSV 기록 + 텔레그램 전송
├── backtest_ml.py          # 45일 분봉 ML 백테스트
├── morning_briefer.py      # 모닝 브리핑
├── data_fetcher.py         # yfinance 일봉 + KIS 분봉
├── indicators.py           # MA / RSI / 볼린저밴드
├── strategy.py             # MA/RSI 매수·매도 신호
├── notifier.py             # 텔레그램 메시지 빌더
├── news_fetcher.py         # 네이버 뉴스 API
├── naver_finance.py        # 네이버 증권 재무지표 스크래핑
├── conditional_orders.py   # 조건부 주문 (가격/수익률 조건)
├── gpt_agent.py            # GPT 툴 함수 (langchain_agent에서 호출)
├── state.json              # 봇 활성화 게이트
├── trade_history.csv       # 매매 이력
├── ml/
│   ├── features.py         # 피처 엔지니어링 (15종)
│   ├── model.py            # XGBoost 학습·예측
│   ├── trainer.py          # 전종목 일괄 학습 + 매일 재학습 (5년치)
│   └── models/             # 학습된 모델 pkl 파일
├── signals/
│   ├── scanner.py          # 기술적 트리거 + ML 예측 (블랙리스트 적용)
│   ├── krx_universe.py     # KRX 전체 종목 1차 스크리닝
│   ├── us_universe.py      # S&P 500 전체 종목 1차 스크리닝 (ET 거래량 환산)
│   └── alert.py            # 급등주 신호 알림 메시지
├── portfolio/
│   ├── kelly.py            # 켈리 공식 포지션 사이징
│   ├── safe_portfolio.py   # 안전자산 비중 추적
│   └── rebalancer.py       # 몬테카를로 리밸런싱
├── logs/
│   └── trader.log
├── com.quant.trader.plist
├── com.quant.telegrambot.plist
└── com.quant.dashboard.plist
```

---

## 백테스트 결과 (참고)

**45일 분봉 기반 백테스트** (2026-04-08 ~ 2026-05-23, 미국 S&P 500)

| 종목 | 거래 | 승률 | 손익비 | 평균 수익 |
|------|------|------|--------|---------|
| WDAY | 8건 | 75.0% | 4.10 | +7.01% |
| INTU | 8건 | 62.5% | 1.98 | +4.06% |
| IBM  | 6건 | 50.0% | 28.66 | +6.41% |
| DECK | 4건 | 50.0% | 4.99 | +3.68% |

**종합: 26건 | 승률 61.5% | 평균 +5.29%**

> 분봉으로 장중 신호 감지 → 다음 봉 시가 매수 → 7거래일 후 청산

---

## 설치

```bash
git clone https://github.com/sonjong980304-tech/quant_trader-.git
cd quant_trader
bash install.sh
```

---

## API 키 설정 (.env)

```
KIS_APP_KEY=...
KIS_APP_SECRET=...
KIS_ACCOUNT_NO=...        # 계좌번호 (예: 12345678-01)
KIS_MOCK=true             # 모의투자: true / 실투자: false
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
OPENAI_API_KEY=...
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...
```

---

## 실행

```bash
# ML 모델 학습 (최초 1회 또는 /trainmodel 명령)
python3 ml/trainer.py

# 장중 스케줄러 (07:30 자동 재학습 포함)
python3 runner.py

# 텔레그램 봇
python3 telegram_bot.py

# 45일 분봉 백테스트
python3 backtest_ml.py
```

---

## 주의사항

1. `.env` 파일은 절대 GitHub에 커밋하지 마세요.
2. KIS_APP_KEY 미설정 시 자동으로 시뮬레이션 모드로 동작합니다.
3. ML 모델은 최초 실행 전 반드시 `/trainmodel` 또는 `python3 ml/trainer.py`로 학습이 필요합니다.
4. 모의투자로 충분히 검증 후 실투자로 전환하세요.

---

## 라이선스

MIT License — 개인 교육 및 연구 목적으로만 사용하세요.
