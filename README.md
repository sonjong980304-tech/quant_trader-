# 퀀트 자동매매 시스템

**🌐 Language:** 한국어 | [English](README.en.md)

> **투자 책임 고지**: 이 프로그램은 교육 및 연구 목적으로 제작되었습니다.  
> 실제 투자 손익에 대한 책임은 전적으로 사용자 본인에게 있습니다.  
> 과거 성과가 미래 수익을 보장하지 않습니다.

---

## 프로젝트 개요

KRX 한국 주식 시장을 대상으로 두 개의 독립 에이전트(Mean Reversion + Trend Following)를 슬롯 분리 방식으로 병렬 운용하는 자동매매 시스템입니다. 텔레그램 봇을 통해 신호·청산·리포트를 실시간으로 수신합니다.

---

## 현재 운영 상태

```
LIVE_TRADING = False  (페이퍼 트레이딩 중)
에이전트: reversion (ML 기반) + trend following (규칙 기반)
슬롯: reversion 10종목 + trend 10종목 = 총 20종목 동시 보유 가능
페이퍼 테스트 기간: 2026-06-19 시작 (2주 목표)
```

---

## 전략 아키텍처

### 에이전트 1 — Mean Reversion (ML 기반)

- XGBoost + Platt Scaling (CalibratedClassifierCV)
- 트리플 배리어 레이블링: TP=+15%, SL=-8%, hold=10일
- Walk-Forward Expanding Window (3-Fold WF)
- OOF AUC: 0.5270 (전체 OOF) / 0.6591 (valid 2026) (8 피처)
- 피처: atr_pct, kospi_relative_20d, beta_60d, ma200_deviation, ret_60d, ret_20d, high52_pct, kospi_relative_5d

### 에이전트 2 — Trend Following (규칙 기반)

- ADX≥25 + MA 정배열(5>20>60>200) + 거래량>1.3x
- ATR 기반 trailing stop: 2.0×ATR
- MA20 하향 이탈 시 청산

### 포트폴리오 운용

- 슬롯 분리: reversion 10 / trend 10 (서로 침범 불가)
- 포지션 사이징: 하프켈리 + ATR (각 에이전트별 독립 계산)
- 최대 1종목 비중: 20%
- 레짐 필터: trend 에이전트 전용 — KOSPI 종가 > KOSPI MA200일 때만 진입
  (reversion 에이전트는 하락장에서도 과매도 반등 포착 목적으로 필터 없음 / SL -8% 자체 손절로 하방 리스크 관리)

---

## 전략 설계 근거 (논문 기반)

### Mean Reversion

- De Bondt & Thaler (1985), "Does the Stock Market Overreact?", Journal of Finance — 투자자 과잉반응으로 급락 종목 반등 실증. reversion 전략 학술 근거.
- Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning", Review of Financial Studies — XGBoost 등 비선형 트리 모델이 수익률 예측에서 선형 모델 압도 실증.
- arXiv:2601.19504 (2026), "Generating Alpha: A Hybrid AI-Driven Trading System", Springer LNNS — RSI/볼린저밴드 평균회귀 + XGBoost + 레짐 필터 결합, 24개월 +135% 달성.
- López de Prado (2018), "Advances in Financial Machine Learning", Wiley — Triple-Barrier 라벨링 방법론 출처. 시간 배리어·TP·SL을 결합해 비선형 레이블을 생성, 금융 ML의 표준 기법으로 자리잡음.

### Trend Following

- Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers", Journal of Finance — 모멘텀 전략 수익성 최초 실증.
- Moskowitz, Ooi & Pedersen (2012) — 추세 추종 형성/보유 기간 최적화 연구.

---

## 논문 대비 차별점

| 항목 | 논문(arXiv:2601.19504) | 본 시스템 |
|------|----------------------|---------|
| 라벨링 | 내일 방향 (단순) | 트리플 배리어 (정교) |
| 검증 방식 | 단순 7:3 분할 | Walk-Forward (시계열 엄밀) |
| 확률 보정 | 없음 | Platt Scaling |
| 평가 지표 | 정확도 63% | AUC (불균형에 강함) |
| 주문 방식 | 시장가 | 익일 시초가 지정가 |
| 대상 시장 | S&P 500 | KRX 한국 주식 |
| 에이전트 | 단일 전략 | reversion + trend 이중 |

---

## 백테스트 결과

### 슬롯 설정별 비교 (2024-01-01 ~ 2026-06-19)

| 설정 | 수익률 | 샤프 | MDD | 거래 수 |
|------|--------|------|-----|---------|
| 공유 10슬롯 (기존) | +84.67% | 1.210 | -25.69% | 1,284건 |
| 분리 5+5 | +130.33% | 1.933 | -15.62% | 493건 |
| **분리 10+10 (채택)** | **+159.31%** | **1.847** | **-20.01%** | **970건** |

### 에이전트별 단독 성과 (2024~2026.6)

| 에이전트 | 수익률 | 샤프 | MDD | 거래 수 | 승률 |
|---------|--------|------|-----|---------|------|
| Reversion (ML) | +29.09% | 0.745 | -18.54% | 469건 | 48.8% |
| Trend Following | +157.87% | 1.928 | -15.59% | 501건 | 43.5% |

### 분리 10+10 전체 지표

- 총 수익률: +159.31% / 샤프: 1.847 / MDD: -20.01% / 거래: 970건 / 승률 46.1% / 손익비 2.02

### 연도별 수익률 (분리 10+10)

| 연도 | 합산 | Reversion | Trend |
|------|------|-----------|-------|
| 2024 | +17.24% | +13.15% | +15.35% |
| 2025 | +47.18% | +1.93% | +45.09% |
| 2026 | +47.68% | +10.10% | +52.87% |

에이전트 비율: reversion 469건 (48%) / trend 501건 (52%)
월별 상관계수 (reversion vs trend): 0.309
현금 비율 (분리 10+10): 2024 31.7%, 2025 48.0%, 2026 32.0%

---

## Trend 에이전트 그리드서치 결과 (27조합)

파라미터: ADX임계값 [20,25,30] × 트레일링스탑 [1.5,2.0,2.5 ATR] × 거래량 [1.0,1.3,1.5x]

유효 조합 상위 5개:

| ADX | Trail | Vol | 수익률 | 샤프 | MDD | 거래 |
|-----|-------|-----|--------|------|-----|------|
| ≥25 | 2.0x | 1.3x | +144.72% | 1.586 | -15.98% | 662건 |
| ≥30 | 2.5x | 1.3x | +143.52% | 1.574 | -14.43% | 542건 |
| ≥25 | 2.5x | 1.3x | +125.73% | 1.459 | -16.94% | 565건 |
| ≥30 | 2.0x | 1.3x | +118.46% | 1.463 | -13.83% | 648건 |
| ≥25 | 2.5x | 1.0x | +114.27% | 1.468 | -19.62% | 615건 |

**채택: ADX≥25 / trail=2.0ATR / vol>1.3x**

---

## Reversion 에이전트 피처 중요도 (8피처)

| 순위 | 피처 | 중요도 |
|------|------|--------|
| 1 | atr_pct | 0.1709 |
| 2 | ret_20d | 0.1348 |
| 3 | kospi_relative_20d | 0.1296 |
| 4 | beta_60d | 0.1253 |
| 5 | ma200_deviation | 0.1236 |
| 6 | high52_pct | 0.1234 |
| 7 | ret_60d | 0.1191 |
| 8 | kospi_relative_5d | 0.0732 |

Walk-Forward AUC (OOF): 0.5270 (TP=15%/SL=8%/hold=10d)

---

## 전략 상세

### Reversion 에이전트 — 신호 파이프라인 (3단계)

**1단계: KRX 유니버스 스크리닝**

FinanceDataReader로 KOSPI+KOSDAQ 전종목 스캔 → 등락률 > 0% + 거래대금 상위 100개 필터링

**2단계: 기술적 트리거 탐지**

| 신호 | 조건 |
|------|------|
| BB하단반등 | 종가가 볼린저밴드 하단 이탈 후 재진입 |
| RSI과매도탈출 | RSI 30 이하에서 30 돌파 |
| 이격도저점 | EMA20 대비 -5% 이하 이격 |

**3단계: XGBoost ML 예측**

피처 (8종): `atr_pct`, `kospi_relative_20d`, `beta_60d`, `ma200_deviation`, `ret_60d`, `ret_20d`, `high52_pct`, `kospi_relative_5d`

Triple-Barrier 라벨링 (López de Prado):

| 배리어 | 조건 | 결과 |
|--------|------|------|
| 상단 TP | 장중 High ≥ 진입가 × 1.15 (+15%) | label=1 (성공) |
| 하단 SL | 장중 Low ≤ 진입가 × 0.92 (−8%) | label=0 (실패) |
| 시간 | 10거래일 경과 후 종가 기준 | 종가 ≥ 진입가 → 1, 미만 → 0 |

### Trend 에이전트 — 진입 조건

| 조건 | 기준 |
|------|------|
| ADX | ≥ 25 |
| MA 정배열 | MA5 > MA20 > MA60 > MA200 |
| 거래량 | 20일 평균 × 1.3배 이상 |
| 레짐 필터 | KOSPI 종가 > KOSPI MA200 (하락장 진입 차단) |
| 청산 | ATR×2.0 trailing stop 또는 MA20 하향 이탈 |

### 포지션 사이징 (하프켈리 + ATR)

```
풀켈리: f* = (p × b - q) / b
하프켈리: f = f* × 0.5

p = ML 예측 승률 (reversion) / 과거 승률 (trend)
b = 평균 수익 / 평균 손실 (손익비)
```

```
리스크동등화 qty = 총자산 × 1% ÷ (2 × ATR(14))
최종 qty = min(하프켈리 qty, 리스크동등화 qty)
```

---

## 자동화 스케줄

| 시간 | 동작 |
|------|------|
| 07:30 (영업일) | 등락률 > 0% + 거래대금 상위 100개 유니버스 ML 모델 병렬 재학습 |
| 08:00 (영업일) | 모닝 브리핑 — AI 시황 + 뉴스 |
| 09:00 (영업일) | KR 예약 주문 실행 — EOD 신호 기반 익일 시초가 매수 |
| **09:05 (영업일)** | **KR 페이퍼 시초가 확정** — `update_entry_prices("KR")` |
| 5분 간격 (장중) | ML 포지션 익절·손절·강제 청산 체크 + 페이퍼 TP/SL 평가 |
| **15:31 (영업일)** | **EOD 신호 스캔** — Close 확정 후 완성 일봉으로 신호 탐지 → 익일 시초가 예약 |
| **15:30 (영업일)** | **KR EOD 평가** — trade_days+1 + TP/SL 체크 |
| **15:35 (매일)** | **KR 페이퍼 트레이딩 일일 리포트** (텔레그램 전송) |
| 일요일 20:00 | 페이퍼 트레이딩 주차별 집계 |

---

## 봇 활성화 게이트

기존 보유 종목이 있을 경우, 해당 종목을 **직접 모두 매도**해야 자동매매가 시작됩니다.

```
state.json: {"bot_active": false, "legacy_tickers": ["XXXX"], ...}
  ↓ 직접 매도 완료
state.json: {"bot_active": true, ...}  →  자동매매 시작
```

**텔레그램 제어**

| 명령 | 동작 |
|------|------|
| `/stop` | 자동매매 중단 |
| `/start` | 자동매매 재개 |

---

## 페이퍼 트레이딩 (`paper_trader.py`)

실거래 전 2주 페이퍼 검증 단계. `LIVE_TRADING=False` 상태에서만 동작하며 실 API 호출 없음.

**슬롯 분리 운용**
- reversion 전용 10슬롯 / trend 전용 10슬롯 완전 분리
- `can_add_position(agent)` 함수로 진입 전 슬롯 여유 확인

**Circuit Breaker 조건 (P3)**

| CB | 조건 | 발동 기준 |
|----|------|---------|
| CB1 | 페이퍼 EV | n≥30 시 EV ≤ −0.5% |
| CB2 | CI 하단 | n≥50 시 95% CI 하단 < −1.0% |
| CB3 | 연속 손실 | 최대 연속 손실 ≥ 8건 |
| CB4 | 백테스트 갭 | n≥30 시 페이퍼 EV − 백테스트 EV ≤ −1.0%p |
| CB5 | 슬리피지 | 실측 평균 슬리피지 > 0.50% |
| CB6 | AUC | 분기 평균 AUC < 0.45 |

**P4 실거래 게이트 기준**

| 항목 | 기준 |
|------|------|
| 페이퍼 운영 기간 | ≥ 60 거래일 |
| 누적 청산 건수 | ≥ 50건 |
| 세후 EV | ≥ +0.30% |
| 95% CI 하단 | > 0% |
| 승률 | ≥ 52% |
| 실측 슬리피지 | < 0.40% |
| 종목 집중도 | < 30% |
| 연속 손실 최대 | ≤ 5건 |
| 레짐 AUC 평균 | ≥ 0.55 |

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
| win_prob | 매수 시점 ML 예측 승률 (%) |
| avg_win_pct | 모델 학습 기준 평균 수익률 (%) |
| avg_loss_pct | 모델 학습 기준 평균 손실률 (%) |
| model_auc | 매수 시점 모델 OOF AUC |

---

## GPT AI 어시스턴트 (LangGraph ReAct)

`create_react_agent` (langgraph.prebuilt) + `MemorySaver` checkpointer로 유저별 `thread_id` 대화 이력을 분리 관리합니다.

| 툴 | 용도 |
|----|------|
| `get_naver_finance` | 한국 주식 재무지표 (PER, PBR, EPS 등) |
| `get_naver_news` | 네이버 최신 뉴스 검색 |
| `get_stock_signal` | 기술적 지표 + 매수/매도 신호 분석 |
| `get_historical_price` | 특정 날짜 종목 종가 조회 |
| `get_account_balance` | 국내 잔고 조회 |
| `get_portfolio_status` | reversion/trend 슬롯 현황 |
| `set_conditional_order` | 조건부 주문 등록 |
| `list_conditional_orders` | 조건부 주문 목록 조회 |
| `cancel_conditional_order` | 조건부 주문 취소 |
| `list_trade_records` | 매매 이력 조회 (open/closed/all) |
| `edit_trade_record` | 매매 이력 수정 |

---

## 텔레그램 명령어

| 명령어 | 설명 |
|--------|------|
| 자유 텍스트 | LangChain AI 어시스턴트 자동 답변 |
| `/ask <질문>` | GPT 명시적 질문 |
| `/reset` | 대화 기록 초기화 |
| `/status` | 전 종목 신호 조회 |
| `/balance` | 국내 잔고 조회 |
| `/portfolio` | 포트폴리오 현황 |
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
       │    │     langchain_agent.py  (LangGraph ReAct + gpt-5.5)    │
       │    │   MemorySaver checkpointer — thread_id 유저별 대화 분리  │
       │    └───────────────────────────────────────────────────────┘
       │
       ├── [07:30] ml/trainer.py — 등락률 > 0% + 거래대금 상위 100개 유니버스 XGBoost 병렬 재학습
       │
       ├── [08:00] morning_briefer.py — 보유종목 뉴스 + 시황
       │
       ├── [5분 간격] check_ml_positions() + _run_paper_evaluate_kr()
       │     → ML 포지션 익절·ATR손절·트레일링스톱·강제 청산 체크
       │     → 페이퍼 포지션 TP/SL 평가 (KR 장중)
       │
       ├── [15:31] scan_growth_signals_eod()
       │     KOSPI MA200 기반 레짐 필터 (trend 전용)
       │     signals/krx_universe.py → FinanceDataReader KOSPI+KOSDAQ 스크리닝
       │     signals/signal_graph.py — 신호 탐지 파이프라인
       │       ├── reversion 에이전트: BB하단반등·RSI과매도탈출·이격도저점
       │       └── trend 에이전트: ADX≥25 + MA정배열 + 거래량>1.3x
       │     → 슬롯 확인 후 페이퍼 기록 또는 pending_orders 등록
       │
       ├── [15:30] _run_paper_evaluate_kr_eod() — KR EOD trade_days+1
       ├── [15:35] paper_trader.daily_report(market="KR") — KR 페이퍼 일일 리포트
       └── [일요일 20:00] paper_trader.weekly_summary()

공통 레이어:
  KIS API (한국 실시간 현재가·주문·잔고)  ·  yfinance ≥1.2
  FinanceDataReader (KRX 유니버스 스크리닝)  ·  ml/models/*.pkl
  trade_history.csv  ·  state.json
```

---

## 설치 및 실행 방법

### 설치

```bash
git clone https://github.com/sonjong980304-tech/quant_trader-.git
cd quant_trader
bash install.sh
```

### API 키 설정 (.env)

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

### 실행

```bash
# ML 모델 학습 (최초 1회 또는 /trainmodel 명령)
python3 ml/trainer.py

# 장중 스케줄러 (07:30 자동 재학습 포함)
python3 runner.py

# 텔레그램 봇
python3 telegram_bot.py

# 45일 분봉 백테스트
python3 backtest_ml.py

# 슬롯 분리 합산 백테스트
python3 combined_backtest.py
```

---

## 디렉토리 구조

```
quant_trader/
├── config.py               # 전략 파라미터 / API 설정
├── stocks.py               # 관심종목 (STOCKS)
├── runner.py               # 스케줄러
├── telegram_bot.py         # 텔레그램 봇
├── langchain_agent.py      # LangGraph ReAct AI 어시스턴트
├── pending_confirmations.py # EOD 매수 신호 확인 대기 목록
├── trader.py               # KIS API (국내)
├── trade_logger.py         # 매매 이력 CSV 기록 + 텔레그램 전송
├── backtest_ml.py          # 45일 분봉 ML 백테스트
├── backtest_walkforward.py # Walk-forward 백테스트 (비용 반영)
├── combined_backtest.py    # 슬롯 분리 합산 백테스트
├── paper_trader.py         # 페이퍼 트레이딩 엔진 (슬롯 분리 10+10, Circuit Breaker)
├── position_manager.py     # ML 포지션 추적 및 봇 활성화 상태 관리
├── trend_agent.py          # Trend Following 에이전트
├── tests/
│   ├── test_triple_barrier.py      # Triple-Barrier 라벨링 단위 테스트
│   ├── test_paper_trader.py        # 페이퍼 트레이딩 엔진 단위 테스트
│   └── test_position_manager.py    # ML 포지션 추적 단위 테스트
├── morning_briefer.py      # 모닝 브리핑 (LangGraph 품질 재시도 루프)
├── data_fetcher.py         # yfinance 일봉 + KIS 분봉
├── indicators.py           # MA / RSI / 볼린저밴드
├── strategy.py             # MA/RSI 매수·매도 신호
├── notifier.py             # 텔레그램 메시지 빌더
├── news_fetcher.py         # 네이버 뉴스 API
├── naver_finance.py        # 네이버 증권 재무지표 스크래핑
├── conditional_orders.py   # 조건부 주문 (가격/수익률 조건)
├── market_calendar.py      # KRX 영업일 캐시
├── market_regime.py        # KOSPI 시장 상황 필터
├── gpt_agent.py            # GPT 툴 함수
├── signals/
│   ├── signal_graph.py     # LangGraph StateGraph 신호 탐지 파이프라인
│   ├── scanner.py          # 기술적 트리거 탐지 + ML 에이전트 평가
│   ├── krx_universe.py     # KRX 전체 종목 1차 스크리닝
│   └── alert.py            # 급등주 신호 알림 메시지
├── state.json              # 봇 활성화 게이트
├── trade_history.csv       # 매매 이력
├── ml/
│   ├── features.py         # 피처 엔지니어링 + Triple-Barrier 라벨링
│   ├── model.py            # XGBoost 학습·예측
│   ├── trainer.py          # KRX 유니버스 병렬 재학습
│   └── models/             # {ticker}_momentum.pkl / {ticker}_reversion.pkl
├── portfolio/
│   └── kelly.py            # 켈리 공식 포지션 사이징
├── logs/
│   └── trader.log
├── com.quant.trader.plist
├── com.quant.telegrambot.plist
└── com.quant.dashboard.plist
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
