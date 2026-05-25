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
- 매일 07:30 **KRX 상위 100 + US 상위 50 유니버스 종목 병렬 재학습** (8스레드, 최신 시장 반영)

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

**학습 데이터**: 5년치 일봉 / 매일 07:30 KRX 100 + US 50 유니버스 종목 병렬 재학습

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

## 알고리즘 선택 근거

### XGBoost + TimeSeriesSplit

#### XGBoost를 선택한 이유

XGBoost(Extreme Gradient Boosting)는 결정 트리를 순차적으로 앙상블하는 부스팅 계열 모델입니다.  
주가 데이터에 적합한 이유는 세 가지입니다.

1. **비선형 패턴 포착** — 거래량 급등·RSI 과매도탈출·볼린저밴드 수축 등 기술적 신호는 선형 관계가 아닙니다. 트리 기반 모델은 이런 임계값(threshold) 조건을 자연스럽게 학습합니다.
2. **피처 스케일 무관** — MA이격도(%), RSI(0~100), 거래량비율(배수)처럼 단위가 제각각인 15개 피처를 정규화 없이 그대로 사용할 수 있습니다.
3. **클래스 불균형 대응** — 7일 후 +3% 이상 수익을 내는 케이스는 전체 데이터의 일부입니다. `scale_pos_weight = (양성 외 비율) / 양성 비율`로 소수 클래스에 가중치를 부여해 편향을 보정합니다.

추가로 매일 07:30에 KRX 상위 100 + US 상위 50 유니버스 종목을 8스레드 병렬로 재학습하는 구조상 학습 속도가 중요한데, XGBoost는 GPU 없이도 150종목을 수십 분 내에 처리할 수 있습니다.

#### TimeSeriesSplit을 선택한 이유

일반적인 k-fold 교차검증은 **데이터를 무작위로 섞어** 학습셋/검증셋을 구성합니다.  
주가 데이터에 이를 적용하면 **미래 데이터로 과거를 예측**하는 데이터 누수(data leakage)가 발생해 검증 지표가 과도하게 낙관적으로 나옵니다.

```
일반 k-fold (잘못된 방식)
  Fold 1:  [──val──][──────train──────][──────train──────]
  Fold 2:  [──────train──────][──val──][──────train──────]
                                          ↑ 미래가 과거 학습에 포함

TimeSeriesSplit (올바른 방식)
  Fold 1:  [──────train──────][──val──]
  Fold 2:  [────────────train────────][──val──]
  Fold 3:  [──────────────────train──────────][──val──]
                     과거 → 미래 방향만 허용
```

`TimeSeriesSplit(n_splits=5)`은 각 fold마다 학습 구간을 순차적으로 확장하면서 검증 구간은 항상 미래에 위치합니다. 실전 운용과 동일한 조건에서 성능을 평가하므로 OOF(Out-of-Fold) 지표가 실제 예측력을 신뢰성 있게 반영합니다.

---

### 마코위츠 효율적 투자선과 몬테카를로 시뮬레이션

#### 마코위츠 효율적 투자선이란

![마코위츠 효율적 투자선](docs/efficient_frontier.png)

해리 마코위츠(Harry Markowitz)의 현대 포트폴리오 이론(MPT, 1952)은 **분산투자로 동일한 기대수익률을 더 낮은 위험으로 달성할 수 있다**는 것을 수학적으로 증명했습니다.

자산들의 기대수익률, 분산, 상관관계를 고려하면 가능한 포트폴리오 집합에서 두 종류의 경계가 존재합니다.

- **최소분산 프론티어**: 각 기대수익률 수준에서 분산이 가장 작은 포트폴리오의 집합
- **효율적 투자선(Efficient Frontier)**: 최소분산 프론티어 중 기대수익률이 더 높은 상반부 — 이 선 위의 포트폴리오만이 합리적 선택입니다

이 중 무위험수익률을 고려했을 때 **샤프비율(초과수익 / 변동성)이 최대**인 접점 포트폴리오가 이론적으로 최적 위험자산 배분입니다.

#### 해석적 풀이 대신 몬테카를로를 사용하는 이유

MPT의 해석적 풀이는 공분산 행렬의 역행렬 계산을 요구합니다. 이론적으로 완전하지만 실제 적용에는 한계가 있습니다.

| 한계 | 내용 |
|------|------|
| 수익률 분포 가정 | 해석적 풀이는 정규분포를 가정하지만, 주가 수익률은 팻테일(fat tail)과 비대칭성을 보입니다 |
| 비선형 제약 | 비중 합계 = 1, 비중 ≥ 0(공매도 금지) 같은 제약을 추가하면 볼록 최적화 문제가 복잡해집니다 |
| 소표본 불안정성 | 공분산 행렬 추정 오차가 역행렬 계산에 증폭되어 극단적 비중이 나올 수 있습니다 |

몬테카를로 시뮬레이션은 **10만 번 무작위 비중을 샘플링**해 각각의 샤프비율을 계산한 뒤 최댓값을 취합니다.

```python
# rebalancer.py 핵심 로직
returns = prices.pct_change().dropna()
mean_ret = returns.mean()
cov = returns.cov()

best_sharpe, best_weights = -np.inf, None
for _ in range(100_000):
    w = np.random.dirichlet(np.ones(n))          # 비중 합계 = 1 자동 보장
    port_ret = mean_ret @ w * 252                # 연환산 수익률
    port_vol = np.sqrt(w @ cov @ w * 252)        # 연환산 변동성
    sharpe   = (port_ret - RISK_FREE_RATE) / port_vol
    if sharpe > best_sharpe:
        best_sharpe, best_weights = sharpe, w    # 효율적 투자선 접점 추적
```

이 방식은 정규분포 가정 없이 실제 수익률 분포를 반영하고, 공매도 금지 제약이 Dirichlet 샘플링으로 자연스럽게 충족되며, 구현 복잡도가 낮아 매월 재실행하기에 적합합니다.

---

### 풀켈리 vs 하프켈리

#### 켈리 공식이란

켈리 공식은 반복적인 베팅에서 **장기 자산의 기하평균 성장률을 최대화**하는 최적 베팅 비율을 도출합니다.

```
풀켈리:   f* = (p × b - q) / b

  p = 승률 (win probability)
  b = 손익비 = 평균 수익률 / 평균 손실률
  q = 1 - p (패배 확률)
```

예를 들어 승률 60%, 손익비 2.0이면 `f* = (0.6 × 2 - 0.4) / 2 = 0.4` → 자산의 40%를 베팅합니다.

#### 풀켈리의 문제점

풀켈리는 이론적으로 최적이지만, **입력값이 정확히 알려져 있다는 전제** 위에 성립합니다.

이 시스템에서 `p`(승률)와 `b`(손익비)는 XGBoost가 과거 데이터에서 학습한 **추정값**입니다. 실제 미래 시장에서 이 수치가 정확히 재현된다는 보장이 없습니다.

| 시나리오 | 결과 |
|----------|------|
| 추정 승률 0.60 → 실제 0.52 | 풀켈리 과다 베팅 → 드로다운 급증 |
| 손익비 과대 추정 | 손실 시 포트폴리오 급감 |
| 연속 손실 구간 | 풀켈리는 기하평균 최적이나 심리적 감내 한계 초과 |

#### 하프켈리를 사용하는 이유

```
하프켈리:  f = f* × 0.5
```

하프켈리는 수학적으로 다음 특성을 가집니다.

- **드로다운** 크기: 풀켈리 대비 약 **75% 수준**으로 감소
- **장기 성장률**: 풀켈리 대비 약 **75% 수준** 유지
- **입력 오차 민감도**: 추정값이 실제와 달라도 손실 폭이 크게 줄어듦

즉, 성장률을 25% 포기하는 대신 드로다운 위험을 25% 줄이는 거래입니다. ML 모델의 추정 오차가 필연적으로 존재하는 환경에서 이 완충은 전략 지속성을 유지하는 데 중요합니다.

실제 적용에는 최대 25% 한도도 추가로 적용됩니다(`min(f_half, 0.25)`).

```python
# kelly.py
f_full = (win_prob * b - q) / b  # 풀켈리
f_half = f_full * 0.5            # 하프켈리 적용
return round(max(0.0, f_half), 4)
```

---

## 자동화 스케줄

| 시간 | 동작 |
|------|------|
| 07:30 (영업일) | KRX 상위 100 유니버스 ML 모델 병렬 재학습 |
| 22:30 / 23:30 (평일) | US 상위 50 유니버스 ML 모델 재학습 (서머/동절기 자동 분기, ET 09:30~10:00 창에서만 실행) |
| 08:00 (영업일) | 모닝 브리핑 (보유종목 뉴스 + 시황) |
| 08:30 (매월 1일 영업일) | 안전자산 몬테카를로 리밸런싱 |
| 5분 간격 (한국장 09:00~15:30) | 분봉 기반 급등주 ML 신호 스캔 |
| 5분 간격 (미국장 ET 09:30~16:00) | S&P 500 분봉 기반 급등주 ML 신호 스캔 |
| 15:00 (영업일) | 일일 기술적 분석 리포트 |

> **영업일 자동 감지**: `market_calendar.py`가 pykrx로 KRX 연간 영업일을 캐시하여 주말 + 공휴일(빨간날) 모두 자동 제외

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
       ├── [07:30] ml/trainer.py — KRX 100+US 50 유니버스 XGBoost 병렬 재학습
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
├── market_calendar.py      # KRX 영업일 캐시 (주말+공휴일 자동 감지)
├── gpt_agent.py            # GPT 툴 함수 (langchain_agent에서 호출)
├── state.json              # 봇 활성화 게이트
├── trade_history.csv       # 매매 이력
├── ml/
│   ├── features.py         # 피처 엔지니어링 (15종)
│   ├── model.py            # XGBoost 학습·예측
│   ├── trainer.py          # KRX+US 유니버스 병렬 재학습 (8스레드, 5년치)
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
| INTU | 8건 | 87.5% | 3.21 | +5.50% |
| WDAY | 8건 | 87.5% | 5.37 | +5.99% |
| DECK | 4건 | 50.0% | 3.94 | +2.71% |
| IBM  | 5건 | 40.0% | 11.94 | +1.93% |

**종합: 25건 | 승률 72.0% | 평균 +4.03%**

> 분봉으로 장중 신호 감지 → 다음 봉 시가 매수 → 아래 조건으로 자동 청산

### 자동 청산 조건 (5분마다 체크)

| 조건 | 기준 | 알림 |
|------|------|------|
| ✅ 익절 | 현재가 ≥ 매수가 × (1 + avg_win) | "익절" |
| 🔴 손절 | 현재가 ≤ 매수가 × 0.93 (-7%) | "손절" |
| ⏰ 기간 청산 | 매수 후 7거래일 경과 | "기간 청산" |

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
