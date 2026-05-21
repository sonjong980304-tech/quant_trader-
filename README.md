# 퀀트 자동매매 시스템

> **투자 책임 고지**: 이 프로그램은 교육 및 연구 목적으로 제작되었습니다.  
> 실제 투자 손익에 대한 책임은 전적으로 사용자 본인에게 있습니다.  
> 과거 성과가 미래 수익을 보장하지 않습니다.

---

## 매수 / 매도 전략

### 거래량 기준
장중 실시간 실행 시 현재까지의 누적 거래량을 경과 시간 비율로 환산한 **하루 예상 거래량** 기준으로 비교합니다.  
`예상 거래량 = 현재 누적 거래량 / (경과 분 / 390분)` | 9:30 이전은 신호 미발생

---

### 매수 신호

| 원칙 | 기반 | 조건 |
|------|------|------|
| **1원칙** 시가돌파 | 분봉 (장중 실시간) | 전일 종가 > MA5 AND 9:00~9:30 저가 < 시가 AND 9:30 이후 현재가 > 시가 돌파 AND 거래량 > 직전 5분봉 평균 × 1.5배 |
| **2원칙** MA사이반등 | 일봉 | 전일 종가가 MA5~MA20 사이 AND 예상 거래량 > 50일 평균 × 1.5배 AND 양봉 |
| **3원칙** MA20아래급등 | 일봉 | 전일 종가 < MA20 AND 예상 거래량 > 50일 평균 × 2.0배 AND (양봉 또는 도지) |

---

### 매도 신호

| 원칙 | 실행 | 조건 |
|------|------|------|
| **1원칙** 급등 후 장대음봉 | 보유량 50% 매도 | 종가 > MA5 AND 예상 거래량 > 50일 평균 × 2.0배 AND 장대음봉 |
| **2원칙** MA5~MA20 사이 이탈 | 전량 매도 | 종가가 MA5~MA20 사이 AND 예상 거래량 > 50일 평균 × 1.5배 AND 음봉 |

---

## 주문 규칙

매수 신호 발생 시 **총 자산(현금 + 평가액)의 40%** 한도로 시장가 매수.  
1주 가격이 40%를 초과하는 경우 **1주** 매수.  
이미 보유 중인 종목은 추가 매수하지 않습니다.

---

## 분봉 거래량 급증 알림

장중 분봉 거래량이 직전 5분봉 평균 대비 **5배 이상** 급증하면,  
해당 종목의 네이버 최신 뉴스 3건을 자동으로 텔레그램으로 전송합니다.  
동일 종목은 30분 이내 중복 알림을 차단합니다.

---

## 일일 기술적 분석 리포트

매일 **15:00**에 전 종목의 기술적 분석 요약을 텔레그램으로 자동 전송합니다.

```
📈 삼성전자
현재가: 75,000원  ▲1.23%
MA5: 74,200 | MA20: 73,100 (우상향↗)
위치: MA5 위 / MA20 위
RSI: 58.3 | 캔들: 양봉
거래량: 1.8배 (50일 평균 대비)
신호: 🟢 매수 (2원칙)
포지션: 미보유
```

---

## GPT AI 어시스턴트

텔레그램에서 자연어로 질문하면 GPT-4.1이 자동으로 답변합니다.  
재무·신호·뉴스가 필요한 경우 아래 툴을 자동 호출해 실시간 데이터를 기반으로 답변합니다.

| 툴 | 용도 | 예시 질문 |
|----|------|----------|
| `get_naver_finance` | 한국 주식 재무지표 + 현재주가 (PER, PBR, EPS, 연도별 예상치 포함) | "삼성전자 26년 예상 EPS 기준 PER 계산해줘" |
| `get_historical_price` | 특정 날짜 종목 종가 조회 | "삼성전자 26년 5월 18일 주가 알려줘" |
| `get_yahoo_finance` | 미국 주식 재무지표 + 연간 실적 시계열 | "NVDA 재무 보여줘" |
| `get_stock_signal` | 매수/매도 원칙별 조건 충족 여부 분석 | "LG전자 왜 매수 안 됐어?" |
| `get_naver_news` | 네이버 최신 뉴스 검색 | "현대차 요즘 이슈 뭐야?" |
| `propose_trade` | 자연어 주문 → 2단계 확인 후 실행 | "현대모비스 1주 사줘" |
| `set_conditional_order` | 조건부 주문 등록 | "삼성전자 7만원 아래 떨어지면 1주 사줘" |
| `list_conditional_orders` | 등록된 조건부 주문 목록 조회 | "조건부 주문 뭐 걸어놨어?" |
| `cancel_conditional_order` | 조건부 주문 취소 | "조건부 주문 전부 취소해줘" |

툴을 사용한 답변에는 **Context Recall 점수**가 자동으로 표시됩니다 (서브에이전트 gpt-4.1-mini 평가).

### 자연어 주문 흐름

```
사용자: "삼성전자 10주 사줘"
  ↓
GPT: ⚠️ 삼성전자(005930) 10주 시장가 매수
     확인하시겠습니까? (예 / 아니오)
  ↓
사용자: "응"
  ↓
✅ 삼성전자(005930) 10주 시장가 매수 주문 완료!
```

### 조건부 주문

특정 가격 또는 수익률 조건이 충족될 때 자동으로 주문이 실행됩니다.  
조건은 JSON 파일로 저장되어 봇 재시작 후에도 유지됩니다.

| 조건 타입 | 설명 | 예시 |
|-----------|------|------|
| `price_below` | 현재가 < 기준가 | "현대모비스 60만원 아래 떨어지면 1주 사줘" |
| `price_above` | 현재가 > 기준가 | "삼성전자 8만원 넘으면 팔아줘" |
| `profit_above` | 수익률 > X% (익절) | "현대모비스 +10% 되면 전량 팔아줘" |
| `profit_below` | 수익률 < X% (손절) | "현대모비스 -5% 되면 손절해줘" |

- runner.py 5분 루프마다 조건 체크 → 충족 시 자동 실행 + 텔레그램 알림
- 정확한 가격 타이밍이 아닌 5분 체크 시점 기준 실행 (소프트웨어 레벨 조건부)

---

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    macOS launchd — 3개 데몬 상시 실행                         │
├──────────────────┬──────────────────────────┬──────────────────────────────┤
│  runner.py       │  telegram_bot.py          │  dashboard.py                │
│  (5분 장중 루프)  │  (사용자 인터페이스)        │  (Streamlit 모니터링)          │
└────────┬─────────┴───────────┬──────────────┴──────────────────────────────┘
         │                     │
         │    ┌────────────────▼─────────────────────────────────────────┐
         │    │              gpt_agent.py  (GPT-4.1)                     │
         │    │                                                           │
         │    │  시스템 프롬프트: 매매 전략 + 실시간 잔고(KIS API)           │
         │    │                                                           │
         │    │  툴 자동 호출                                              │
         │    │  ├─ get_stock_signal ──▶ fetch_ohlcv + generate_signals  │
         │    │  │                       (runner.py와 동일 파이프라인)      │
         │    │  ├─ get_naver_finance ─▶ naver_finance.py (현재주가+연도별EPS) │
         │    │  ├─ get_historical_price ▶ yfinance (특정날짜 종가)        │
         │    │  ├─ get_yahoo_finance ─▶ yfinance (미국 주식)              │
         │    │  ├─ get_naver_news ───▶ news_fetcher.py (네이버 API)      │
         │    │  ├─ propose_trade ────▶ KIS API 실제 주문 (2단계 확인)     │
         │    │  ├─ set_conditional_order ▶ conditional_orders.py 저장    │
         │    │  ├─ list/cancel_conditional_order                        │
         │    │  └─ Context Recall 평가 (gpt-4.1-mini 서브에이전트)        │
         │    └──────────────────────────────────────────────────────────┘
         │
         │    ┌─────────────────────────────────────────────────────────┐
         │    │  /run 명령어 → graph.py (LangGraph 단발성 에이전트)       │
         │    │  fetch_data → calc_indicators → detect_signal           │
         │    │      → (신호 있음) → send_notification → save_log        │
         │    └─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  공통 데이터 레이어                                                            │
│  yfinance (일봉 1년)  ·  KIS API (분봉·현재가·주문·잔고)                       │
│  conditional_orders.json (조건부 주문 영속 저장)                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## LangGraph 실행 흐름 (graph.py)

```
┌─────────────────────────────────────────────────────────────────┐
│                     종목별 순차 실행                              │
│                                                                 │
│  [노드1] 데이터 수집                                             │
│    └─ yfinance로 1년치 OHLCV 수집                               │
│           │                                                     │
│  [노드2] 지표 계산                                               │
│    └─ MA5 / MA20 / RSI14 / 골든크로스 / 데드크로스               │
│           │                                                     │
│  [노드3] 신호 감지                                               │
│    └─ 매수 1~3원칙 / 매도 1~2원칙 평가                           │
│           │                                                     │
│        신호 없음? ──────────────────────────────► END           │
│           │ 신호 있음                                           │
│  [노드4] 알림 전송 + 주문 실행                                   │
│    └─ 텔레그램 알림 → KIS API 시장가 주문                        │
│           │                                                     │
│  [노드5] 로그 저장                                               │
│    └─ logs/trader.log (JSON 포맷, 5MB 롤링)                     │
│           │                                                     │
│          END                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 상태(State) 흐름

```python
TraderState {
    ticker        # 종목 코드
    stock_name    # 종목명
    ohlcv         # OHLCV DataFrame (노드1~3에서 갱신)
    signal        # 최신 신호 딕셔너리 (close, MA5, MA20, RSI, volume...)
    signal_type   # "buy" / "sell_full" / "sell_partial" / "none"
    log_entries   # 로그 항목 목록
    error         # 예외 메시지
}
```

---

## 프로젝트 구조

```
quant_trader/
├── config.py           # 전략 파라미터 / API 설정
├── data_fetcher.py     # yfinance 일봉 + KIS API 분봉 수집
├── indicators.py       # MA / RSI / 거래량 시간 보정
├── strategy.py         # 매수 1~3원칙 / 매도 1~2원칙 신호 생성
├── trader.py           # KIS API 주문 / 잔고 / 포지션 관리
├── notifier.py         # 텔레그램 알림 메시지 빌더
├── news_fetcher.py     # 네이버 뉴스 API 수집
├── naver_finance.py    # 네이버 증권 재무지표 스크래핑
├── conditional_orders.py # 조건부 주문 관리 (가격/수익률 조건, JSON 영속 저장)
├── gpt_agent.py        # GPT-4.1 AI 어시스턴트 (툴 8종 + Context Recall 평가)
├── graph.py            # LangGraph 5노드 에이전트
├── runner.py           # 장중 분봉 기반 5분 간격 실시간 매매 루프
├── telegram_bot.py     # 텔레그램 봇 (자연어 대화 + 명령어)
├── backtest.py         # vectorbt 백테스트
├── dashboard.py        # Streamlit 모니터링 대시보드
├── install.sh          # 자동 설치 스크립트
├── com.quant.trader.plist      # launchd — 매매 루프
├── com.quant.dashboard.plist   # launchd — 대시보드
├── com.quant.telegrambot.plist # launchd — 텔레그램 봇
├── logs/
│   └── trader.log
└── results/
    └── backtest_result.png
```

---

## 설치

```bash
git clone https://github.com/sonjong980304-tech/quant_trader-.git
cd quant_trader
bash install.sh
```

설치 스크립트가 다음을 자동 처리합니다:
- Python 패키지 설치 (`requirements.txt`)
- `.env` 파일 생성 및 API 키 입력 안내
- macOS launchd 자동실행 등록

### 수동 설치

```bash
pip install -r requirements.txt
cp .env.example .env
# .env 파일에 API 키 입력
```

---

## API 키 설정 (.env)

```
KIS_APP_KEY=...
KIS_APP_SECRET=...
KIS_ACCOUNT_NO=...
KIS_MOCK=true          # 모의투자: true / 실투자: false
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
OPENAI_API_KEY=...
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...
```

### 발급 방법

- **KIS API**: [한국투자증권 KIS Developers](https://apiportal.koreainvestment.com) → 앱 신청
- **텔레그램 봇**: `@BotFather` → `/newbot` → Chat ID는 `getUpdates` API로 확인
- **OpenAI**: [platform.openai.com](https://platform.openai.com) → API Keys
- **네이버 뉴스 API**: [developers.naver.com](https://developers.naver.com) → 애플리케이션 등록 → 검색 API

---

## 실행

```bash
# LangGraph 에이전트 1회 실행
python graph.py

# 장중 실시간 매매 루프
python runner.py

# 백테스트 (최근 3년)
python backtest.py

# 대시보드
streamlit run dashboard.py
```

### macOS 자동실행 관리

```bash
# 활성화
launchctl load ~/Library/LaunchAgents/com.quant.trader.plist
launchctl load ~/Library/LaunchAgents/com.quant.telegrambot.plist

# 비활성화
launchctl unload ~/Library/LaunchAgents/com.quant.trader.plist
launchctl unload ~/Library/LaunchAgents/com.quant.telegrambot.plist
```

---

## 텔레그램 명령어

| 명령어 | 설명 |
|--------|------|
| 자유 텍스트 | GPT AI 어시스턴트 자동 답변 |
| `/ask <질문>` | GPT 명시적 질문 |
| `/reset` | 대화 기록 초기화 |
| `/status` | 전 종목 신호 조회 |
| `/balance` | 계좌 잔고 조회 |
| `/run` | LangGraph 에이전트 수동 실행 |
| `/stocks` | 매매 종목 목록 |
| `/addstock 코드 이름` | 종목 추가 |
| `/removestock 코드` | 종목 삭제 |
| `/buy 코드 수량` | 수동 매수 |
| `/sell 코드 수량` | 수동 매도 |
| `/sellall 코드` | 전량 매도 |

---

## 모의 → 실투자 전환

`.env`에서:
```
KIS_MOCK=false
```

> 실투자 전환 전 반드시 모의투자로 충분히 검증하세요.

---

## 주의사항

1. `.env` 파일은 절대 GitHub에 커밋하지 마세요.
2. API 키 없이 실행 시 자동으로 시뮬레이션 모드로 동작합니다.
3. 알고리즘 트레이딩은 예상치 못한 손실이 발생할 수 있습니다.
4. 자연어 주문은 반드시 2단계 확인을 거쳐 실행됩니다.

---

## 라이선스

MIT License — 개인 교육 및 연구 목적으로만 사용하세요.
