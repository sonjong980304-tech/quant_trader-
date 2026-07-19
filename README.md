# 퀀트 자동매매 시스템

🌐 Language: 한국어 | [English](README.en.md)
---

## 개요

KRX 한국 주식 시장을 대상으로 두 개의 독립 에이전트(Mean Reversion + Trend Following)를 슬롯 분리 방식으로 병렬 운용하는 자동매매 시스템입니다. 텔레그램 봇을 통해 신호·청산·리포트를 실시간으로 수신합니다.

---

## 현재 운영 상태

```
LIVE_TRADING = False  (페이퍼 트레이딩 중)
에이전트: reversion (ML 기반) + trend following (규칙 기반)
슬롯: reversion 10종목 + trend 10종목 = 총 20종목 동시 보유 가능
페이퍼 테스트 2차: 2026-07-09 재가동 (급락장 대응 전략 개선 반영, 아래 섹션 참조)
```

---

## 페이퍼 테스트 1차 결과 및 전략 개선 (2026-07-09)

### 1차 페이퍼테스트 결과 (2026-06-19 ~ 2026-07-08)

같은 기간 KOSPI가 **-17% 급락**하면서 두 에이전트 모두 큰 손실을 냈고, reversion 연속 손실이 누적되어 페이퍼테스트를 중단했습니다.

| 에이전트 | 청산 거래 | 승률 | net_pnl 합 |
|------|------|------|------|
| trend | 11건 | 18% | **-78.1%p** |
| reversion | 17건 | 18% | **-67.3%p** |
| 합계 | 28건 | 18% | **-145.3%p** |

**원인 진단:**
- **trend** — 레짐 필터가 `KOSPI 종가 > MA200` 하나뿐이었는데, MA200은 200일 후행 평균이라 지수가 -17% 빠지는 동안에도 전혀 안 깨져 무방비로 진입이 계속됨(전형적 whipsaw).
- **reversion** — 지수 레벨 레짐 필터가 없어 "과매도 반등"을 노리다 추세적 급락(칼날 잡기)에 반복 노출됨. 별도로 학습 피처 12개 중 8개가 permutation importance 검증에서 노이즈로 확인됨(검증 AUC 0.50 수준, 사실상 랜덤).

### 적용한 개선 (2023~2026 4년 walk-forward 백테스트로 검증 후 반영)

**1. trend 레짐 필터 강화** — `KOSPI 종가 > MA200` **AND** `20일 고점 대비 낙폭 > -5%` 오버레이 추가(`market_regime.py`, `runner.py`). 급락 초입에 신규 진입을 차단.

**2. reversion 피처 축소 (12개 → 4개)** — permutation importance 기준 실질 기여가 있는 `kospi_relative_5d`, `candle_body`, `rsi`, `low52_pct`만 남기고 노이즈 피처(`atr_pct` 등) 제거(`ml/features.py`). 축소 모델은 2023~2026 **4개 연도 전부**에서 기존(12개) 대비 개선, 2024년은 손실(-3.5%)에서 흑자(+5.2%)로 전환, 1차 페이퍼 기간(6~7월) 손익은 -134%p → -6%p로 크게 개선됨.

**3. 학습/서빙 파이프라인 버그 수정** — 분기별 자동 재학습(`ml/trainer.py`의 `retrain_daily`)이 피처 축소 상수를 무시하고 레거시 17개 피처로 재학습하던 버그를 발견해 수정. 올바른 경로로 재학습해 프로덕션 모델이 실제로 4피처로 적용됐음을 확인.

#### 🔍 왜 12개 → 4개로 줄였나 — Permutation Importance

**Permutation Importance(순열 중요도)**는 이미 학습이 끝난 모델에서 "피처 하나의 값만 행끼리 무작위로 뒤섞은 뒤" 검증 데이터로 성능을 다시 재보는 방법입니다. 밴드 합주에서 특정 악기 파트만 엉망으로 뒤섞어 틀어보고 곡이 얼마나 망가지는지로 그 악기가 실제로 필요한지 가늠하는 것과 같은 원리입니다.

- 뒤섞었을 때 검증 AUC(오를 종목/내릴 종목을 얼마나 잘 구별하는지 점수)가 많이 떨어짐 → 그 피처는 실제로 예측에 기여하는 **진짜 신호**.
- 뒤섞어도 성능이 그대로거나 **오히려 좋아짐**(중요도가 음수) → 모델이 노이즈를 학습하고 있었다는 뜻으로, 빼는 게 낫다.

트리 모델이 자체 제공하는 `feature_importances_`는 **학습 데이터** 안에서 측정돼 과대평가되기 쉬운 반면, permutation importance는 모델이 한 번도 본 적 없는 **검증 데이터**로 측정하기 때문에 "실전에서도 진짜 쓸모 있는 피처인지"를 더 정직하게 보여줍니다.

원래 reversion 모델은 아래 12개 피처를 썼고, 검증 결과에 따라 4개만 남겼습니다:

| 유지 (4개) | 제거 — permutation importance 음수/노이즈로 판정 (8개) |
|---|---|
| `kospi_relative_5d` (코스피 대비 5일 상대강도) | `ret_5d`, `ret_3d` (단기 수익률) |
| `candle_body` (캔들 몸통 크기) | `bb_pct_20` (볼린저밴드 %B) |
| `rsi` | `bb_std_20` (볼린저밴드 20일 **표준편차** — 변동성) |
| `low52_pct` (52주 저점 근접도) | `atr_pct` (ATR 기반 **변동성** 비율) |
| | `high52_pct` (52주 고점 근접도) |
| | `ema_deviation_20` (EMA20 이격도) |
| | `rsi_oversold` (RSI 과매도 플래그) |

제거된 8개 중 `bb_std_20`·`atr_pct`가 대표적인 "변동성/표준편차" 계열 피처인데, 둘 다 permutation importance가 음수(뒤섞었더니 오히려 검증 AUC가 올라감)로 나와 재도입을 금지했습니다. `tests/test_features.py`가 이 4개 피처셋을 회귀 가드로 고정해, 향후 무심코 노이즈 피처를 되돌리는 것을 막습니다.

> ⚠️ 위 개선은 4년 백테스트로 검증됐지만, reversion 쪽은 검증에 쓴 기간으로 피처를 고르기도 해서 선택 편향 여지가 있습니다. 실전 개선폭은 백테스트보다 보수적일 수 있어 페이퍼 2차 결과로 추가 확인이 필요합니다.

### 현재 상태

**2026-07-09부로 위 개선안을 반영해 페이퍼테스트를 재가동했습니다.** 열린 포지션 없이 클린한 상태로 재시작.

---

## 전략 아키텍처

### 에이전트 1 — Mean Reversion (ML 기반)

- XGBoost (Raw 확률, 보정 없음)
- 트리플 배리어 레이블링: TP=+15%, SL=-8%, hold=10일
- Walk-Forward Expanding Window (4-Fold WF, 2023~2026)
- 유니버스: 시점별 동적 PIT 시총 상위 200 (생존편향 제거)
- 피처(2026-07-09 12개→4개 축소): kospi_relative_5d, candle_body, rsi, low52_pct (아래 "페이퍼 테스트 1차 결과" 참조)

### 에이전트 2 — Trend Following (규칙 기반)

- ADX≥25 + MA 정배열(5>20>60>200) + 거래량>1.3x
- ATR 기반 trailing stop: 2.0×ATR
- MA20 하향 이탈 시 청산

### 포트폴리오 운용

- 슬롯 분리: reversion 10 / trend 10 (서로 침범 불가)
- 포지션 사이징: 하프켈리 + ATR (각 에이전트별 독립 계산)
- 최대 1종목 비중: 20%
- 레짐 필터: trend 에이전트 전용 — KOSPI 종가 > MA200 **AND** 20일 고점 대비 낙폭 > -5%일 때만 진입 (2026-07-09 급락 대응 강화, 아래 "페이퍼 테스트 1차 결과" 참조)
  (reversion 에이전트는 하락장에서도 과매도 반등 포착 목적으로 레짐 필터 없음 / SL -8% 자체 손절로 하방 리스크 관리)

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
| 검증 방식 | 단순 7:3 분할 | Walk-Forward + PIT 유니버스 (시계열 엄밀) |
| 확률 보정 | 없음 | 없음 (Raw XGBoost — 백테스트와 정합) |
| 평가 지표 | 정확도 63% | AUC (불균형에 강함) |
| 주문 방식 | 시장가 | 익일 시초가 지정가 |
| 대상 시장 | S&P 500 | KRX 한국 주식 |
| 에이전트 | 단일 전략 | reversion + trend 이중 |

---

## 백테스트 결과

> 확정 전략: **D — Expanding Window + PIT 200 유니버스**

### 최종 성과 (2023~2026, 워크포워드 4-Fold)

| 항목 | 결과 |
|------|------|
| 총수익률 | **+78.52%** |
| 샤프 | **1.037** |
| MDD | -15.84% |
| 거래 수 | 1,545건 (reversion 883 / trend 662) |
| 승률 | 43.3% |
| 손익비 | 1.74 |

### 연도별 수익률

| 연도 | 수익률 |
|------|--------|
| 2023 | +13.62% |
| 2024 | +0.92% |
| 2025 | +28.26% |
| 2026 | +18.38% |

### 4-way 방법론 비교 검증

| 방식 | 수익률 | 샤프 | MDD |
|------|--------|------|-----|
| A) Expanding + 정적 200 | +100.35% | 1.196 | -13.43% |
| **D) Expanding + PIT 200 (채택)** | **+78.52%** | **1.037** | -15.84% |
| B) Rolling 3년 + 정적 200 | +91.01% | 1.148 | -15.88% |
| C) Rolling 3년 + PIT 200 | +72.70% | 1.024 | -15.08% |

생존편향 제거 효과 (A→D): **-21.8%p** / 학습방식 변경 효과 (A→B): -9.3%p

---

## 백테스트 신뢰성 — 편향 발견 및 정량 보정

초기 백테스트는 +159%라는 높은 수치를 보였으나, 아래 편향을 체계적으로 발견하고 정량 보정했습니다.

### 발견된 편향과 보정

**1. 생존편향 (+21.8%p 과대 추정)**
- 초기: 2026년 현재 시총 상위 200개로 2024~2026 과거 백테스트 → 미래 정보 사용
- 보정: 각 시점 기준 동적 PIT(Point-in-Time) 유니버스로 교체 → -21.8%p 보정

**2. 검증기간 편향 (2024~2026 → 2023~2026)**
- 초기: 2024년부터 검증 (표본 부족, 호황기만 포함)
- 보정: 2023년까지 검증 구간 확장, 학습 데이터도 2020년부터 사용

**3. 학습방식 비교 검증 (Rolling vs Expanding)**
- Rolling 3년 vs Expanding window를 4-way 비교로 실증 검증
- 결과: Expanding 채택 — Rolling은 오래된 패턴을 버려 데이터 손실로 -9.3%p 열위

**4. 확률 보정(Platt Scaling) 제거**
- 백테스트는 규칙 기반 트리거 신호이고 실제 운용은 Platt Scaling 적용 → 불일치 발견
- Raw XGBoost 확률로 통일하여 백테스트-실운용 정합성 확보

> 수익률의 크기보다 **편향을 발견하고 정량적으로 보정한 과정**이 이 시스템의 핵심 신뢰성 근거입니다.
> 보정 후 연평균 수익률 ~+20%, 샤프 1.04가 현실적 기대치입니다.

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

## Reversion 에이전트 피처 중요도 (8피처, 최초 백테스트 기준 — 이후 4개로 추가 축소됨, "페이퍼 테스트 1차 결과" 섹션 참조)

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

1단계: KRX 유니버스 스크리닝

FinanceDataReader로 KOSPI+KOSDAQ 전종목 스캔 → 등락률 > 0% + 거래대금 상위 100개 필터링

2단계: 기술적 트리거 탐지

| 신호 | 조건 |
|------|------|
| BB하단반등 | 종가가 볼린저밴드 하단 이탈 후 재진입 |
| RSI과매도탈출 | RSI 30 이하에서 30 돌파 |
| 이격도저점 | EMA20 대비 -5% 이하 이격 |

3단계: XGBoost ML 예측

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
| 07:30 (1/4/7/10월 1일 또는 직후 첫 영업일) | **분기별 ML 모델 재학습** — Expanding Window + PIT 시총 상위 200 유니버스 (reversion XGBoost만 재학습, trend는 규칙 기반) |
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

텔레그램 제어

| 명령 | 동작 |
|------|------|
| `/stop` | 자동매매 중단 |
| `/start` | 자동매매 재개 |

---

## 페이퍼 트레이딩 (`paper_trader.py`)

실거래 전 2주 페이퍼 검증 단계. `LIVE_TRADING=False` 상태에서만 동작하며 실 API 호출 없음.

슬롯 분리 운용
- reversion 전용 10슬롯 / trend 전용 10슬롯 완전 분리
- `can_add_position(agent)` 함수로 진입 전 슬롯 여유 확인

Circuit Breaker 조건 (P3)

| CB | 조건 | 발동 기준 |
|----|------|---------|
| CB1 | 페이퍼 EV | n≥30 시 EV ≤ −0.5% |
| CB2 | CI 하단 | n≥50 시 95% CI 하단 < −1.0% |
| CB3 | 연속 손실 | 최대 연속 손실 ≥ 8건 |
| CB4 | 백테스트 갭 | n≥30 시 페이퍼 EV − 백테스트 EV ≤ −1.0%p |
| CB5 | 슬리피지 | 실측 평균 슬리피지 > 0.50% |
| CB6 | AUC | 분기 평균 AUC < 0.45 |

P4 실거래 게이트 기준

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
       ├── [07:30, 분기별] ml/trainer.py — Expanding Window + PIT 시총 상위 200 XGBoost 재학습 (1/4/7/10월 1일)
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
pip install -r requirements.txt
cp .env.example .env   # 아래 "API 키 설정"에 맞춰 값 입력
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
├── config.py                    # 전략 파라미터 / API 설정 (공용, 루트 유지)
├── runner.py                    # EOD 스케줄러 (launchd: com.quant.trader, 루트 유지)
├── telegram_bot.py              # 텔레그램 봇 (launchd: com.quant.telegrambot, 루트 유지)
│
├── core/                        # 매매 실행 · 포지션 · 주문
│   ├── position_manager.py      # 실매매 ML 포지션 추적 + 봇 활성화 게이트
│   ├── paper_trader.py          # 페이퍼 트레이딩 엔진 (슬롯 10+10, Circuit Breaker)
│   ├── trader.py                # KIS API (국내)
│   ├── trade_logger.py          # 매매 이력 CSV 기록 + 텔레그램 전송
│   ├── conditional_orders.py    # 조건부 주문 (가격/수익률 조건)
│   ├── pending_orders.py        # 시초가 예약 매수 대기열
│   └── pending_confirmations.py # EOD 매수 신호 확인 대기 목록
│
├── strategy/                    # 신호 · 지표 · 시장 레짐
│   ├── strategy.py              # Reversion 트리거 탐지 (신호 생성)
│   ├── indicators.py            # MA / RSI / 볼린저밴드 등 기술적 지표
│   ├── trend_agent.py           # Trend Following 에이전트
│   ├── market_regime.py         # KOSPI 시장 레짐 필터
│   └── market_calendar.py       # KRX 영업일 캐시
│
├── data/                        # 시세 · 재무 · 뉴스 수집
│   ├── data_fetcher.py          # yfinance 일봉 + KIS 분봉
│   ├── naver_finance.py         # 네이버 증권 재무지표 스크래핑
│   └── news_fetcher.py          # 네이버 뉴스 API
│
├── interface/                   # 외부 인터페이스 (알림 · AI)
│   ├── notifier.py              # 텔레그램 메시지 빌더 / 전송
│   └── langchain_agent.py       # LangGraph ReAct AI 어시스턴트
│
├── backtest/                    # 백테스트
│   ├── backtest_ml.py           # 45일 분봉 ML 백테스트
│   ├── backtest_walkforward.py  # Walk-forward 백테스트 (비용 반영)
│   └── combined_backtest_v2.py  # Rolling 3년 × PIT 유니버스 비교 백테스트
│
├── scripts/                     # 수동 실행 스크립트
│   └── catchup_eod.py           # 놓친 KR 오후 EOD 작업 야간 수동 실행
│
├── signals/                     # 신호 탐지 파이프라인
│   ├── signal_graph.py          # LangGraph StateGraph 신호 탐지
│   ├── scanner.py               # 기술적 트리거 탐지 + ML 에이전트 평가
│   ├── krx_universe.py          # KRX 종목 스크리닝
│   ├── us_universe.py           # 미국 종목 유니버스
│   └── alert.py                 # 급등주 신호 알림 메시지
│
├── ml/                          # 머신러닝
│   ├── features.py              # 피처 엔지니어링 + Triple-Barrier 라벨링
│   ├── model.py                 # XGBoost 학습 · 예측
│   ├── regime_model.py          # 레짐 판정 모델
│   ├── trainer.py               # KRX 유니버스 병렬 재학습
│   └── models/                  # {ticker}_momentum.pkl / {ticker}_reversion.pkl
│
├── news_briefing/               # 뉴스 브리핑 파이프라인 (수집→선별→작성→검증)
│   └── service.py               # run_morning / run_evening 진입점 (외 12개 모듈)
│
├── portfolio/
│   └── kelly.py                 # 켈리 공식 포지션 사이징
│
├── dashboard/                   # Streamlit 대시보드
│   ├── app.py                   # 메인 앱 (페이퍼 / 실매매 탭)
│   ├── data_loader.py           # JSON/CSV 데이터 읽기
│   ├── kis_live.py              # KIS 실시간 현재가 (10초 캐싱)
│   └── charts.py                # plotly 그래프 모음
│
└── tests/                       # 단위 테스트 (파일별 개별 실행 권장 — sys.modules mock 격리)
```

> **루트 유지 파일**: `runner.py`·`telegram_bot.py`는 launchd 데몬이 직접 참조하고, `config.py`는
> 여러 모듈이 공유하는 설정이라 폴더로 옮기지 않습니다. 나머지 매매/전략/데이터/백테스트 모듈은
> 역할별 패키지(`core`·`strategy`·`data`·`interface`·`backtest`·`scripts`)로 분리했습니다.
> 하위 폴더의 백테스트·스크립트를 직접 실행할 때는 저장소 루트에서 `python -m backtest.backtest_ml`
> 처럼 모듈 형태로 실행하거나(권장), 각 파일 상단의 repo-root sys.path 부트스트랩이 경로를 보정합니다.

---

## 📊 Streamlit 대시보드

페이퍼 트레이딩과 실제 매매를 탭으로 분리해 시각화하는 웹 대시보드입니다.
봇 데이터(JSON/CSV)를 **읽기 전용**으로 사용하며, 봇 코드에는 영향을 주지 않습니다.

### 실행

```bash
# 의존성 설치 (최초 1회)
pip install -r requirements.txt   # streamlit, plotly 포함

# 대시보드 실행 (봇과 동일한 파이썬 환경에서)
python3 -m streamlit run dashboard/app.py
# 또는
streamlit run dashboard/app.py
```

실행 후 브라우저에서 `http://localhost:8501` 접속.

### 구성

| 파일 | 역할 |
|------|------|
| `dashboard/app.py` | 메인 앱 (페이퍼 / 실매매 탭) |
| `dashboard/data_loader.py` | 기존 JSON/CSV 데이터 읽기 |
| `dashboard/kis_live.py` | KIS 실시간 현재가 조회(10초 캐싱) |
| `dashboard/charts.py` | plotly 그래프 모음 |

- **탭 A (페이퍼)**: 전종목 신호 테이블(필터/정렬), 누적수익 곡선, 에이전트별·트리거별 성과
- **탭 B (실매매)**: 보유 포지션 실시간 손익(KIS 라이브, 손절 근접 ⚠️), 매매 이력, 누적 실현손익
- 사이드바에서 **30초 자동 새로고침** 토글 가능

> 참고: 페이퍼 신호의 AUC는 저장되지 않아 모델 메타값으로 표시하며, 실매매 CSV에는
> 에이전트/트리거 컬럼이 없어 `strategy`로 그룹핑합니다.

---

## 주의사항

1. `.env` 파일은 절대 GitHub에 커밋하지 마세요.
2. KIS_APP_KEY 미설정 시 자동으로 시뮬레이션 모드로 동작합니다.
3. ML 모델은 최초 실행 전 반드시 `/trainmodel` 또는 `python3 ml/trainer.py`로 학습이 필요합니다.
4. 모의투자로 충분히 검증 후 실투자로 전환하세요.

---

## 라이선스

MIT License — 개인 교육 및 연구 목적으로만 사용하세요.
