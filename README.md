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

| 원칙 | 조건 |
|------|------|
| **1원칙** 5일선 단기 돌파 | 전일 종가 > MA5 AND 당일 시가 < MA5 AND 당일 종가 > MA5 + 양봉 |
| **2원칙** MA5~MA20 사이 반등 | 종가가 MA5~MA20 사이 AND 예상 거래량 > 50일 평균 × 1.5배 AND 양봉 |
| **3원칙** 거래량 급증 반등 | 예상 거래량 > 50일 평균 × 2.5배 AND (양봉 또는 도지형) |

> 1원칙은 분봉 기반 장중 실시간 감지도 병행 (9:00~9:30 저가가 MA5 아래 + 시가 상향 돌파 + 거래량 조건)

---

### 매도 신호 (모두 전량 매도)

| 원칙 | 조건 |
|------|------|
| **1원칙** 급등 후 장대음봉 | 종가 > MA5 AND 예상 거래량 > 50일 평균 × 2.5배 AND 장대음봉 |
| **2원칙** MA5~MA20 사이 이탈 | 종가가 MA5~MA20 사이 AND 예상 거래량 > 50일 평균 × 1.5배 AND 음봉 |

---

## 주문 규칙

매수 신호 발생 시 **가용 현금의 40%** 한도로 시장가 매수.  
1주 가격이 40%를 초과하는 경우 **1주** 매수.

---

## 분봉 거래량 급증 알림

장중 분봉 거래량이 직전 5분봉 평균 대비 **2배 이상** 급증하면,  
해당 종목의 네이버 최신 뉴스 3건을 자동으로 텔레그램으로 전송합니다.  
동일 종목은 30분 이내 중복 알림을 차단합니다.

---

## 일일 기술적 분석 리포트

매일 **15:00**에 전 종목의 기술적 분석 요약을 텔레그램으로 자동 전송합니다.

```
📊 일일 기술적 분석 리포트
2026-05-21 15:00 기준

📈 삼성전자
현재가: 75,000원  ▲1.23%
MA5: 74,200 | MA20: 73,100 (우상향↗)
위치: MA5 위 / MA20 위
RSI: 58.3 | 캔들: 양봉
거래량: 1.8배 (50일 평균 대비)
신호: 🟢 매수 (2원칙)
포지션: 미보유
```

종목당 1건씩 개별 메시지로 전송되며, 보유 중인 종목은 매수가와 현재 수익률도 표시됩니다.

---

## 전체 아키텍처

```
[ macOS launchd — 평일 09:00 자동 실행 ]
                    │
                    ▼
         [ runner.py — 장중 매매 루프 ]          [ graph.py — LangGraph 에이전트 ]
                    │                                         │
         분봉 기반 실시간 매매                        일봉 기반 1회 실행
         거래량 급증 시 뉴스 알림
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
│  [노드4] 리스크 체크                                             │
│    └─ 당일 중복 신호 필터 / 연속 매수 과열 경고                   │
│           │                                                     │
│        중복/차단? ──────────────────────────────► END           │
│           │ 통과                                                │
│  [노드5] 알림 전송 + 주문 실행                                   │
│    └─ 텔레그램 알림 → KIS API 시장가 주문                        │
│           │                                                     │
│  [노드6] 로그 저장                                               │
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
    risk_warning  # 과열 경고 메시지 (노드4)
    error         # 예외 메시지
}
```

---

## 프로젝트 구조

```
quant_trader/
├── config.py          # 전략 파라미터 / API 설정
├── stocks.py          # 매매 종목 목록 (로컬 전용 — .gitignore)
├── data_fetcher.py    # yfinance OHLCV 수집
├── indicators.py      # MA / RSI / 거래량 시간 보정
├── strategy.py        # 매수 1~3원칙 / 매도 1~2원칙 신호 생성
├── news_fetcher.py    # 네이버 뉴스 수집
├── graph.py           # LangGraph 6노드 에이전트
├── runner.py          # 장중 분봉 기반 실시간 매매 루프
├── trader.py          # KIS API 주문 / 잔고 / 포지션 관리
├── backtest.py        # vectorbt 백테스트
├── dashboard.py       # Streamlit 모니터링 대시보드
├── notifier.py        # 텔레그램 알림 메시지 빌더
├── telegram_bot.py    # 텔레그램 봇 명령 처리
├── install.sh         # 자동 설치 스크립트
├── com.quant.trader.plist     # launchd — 매매 루프
├── com.quant.dashboard.plist  # launchd — 대시보드
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
- macOS launchd 자동실행 등록 (매일 09:00)
- `stocks.py` 템플릿 생성 (종목 목록 직접 입력)

### 수동 설치

```bash
pip install -r requirements.txt
cp .env.example .env
# .env 파일에 API 키 입력

# 종목 목록 설정 (stocks.py는 .gitignore에 포함)
cp stocks.py.example stocks.py
# stocks.py에서 STOCKS 딕셔너리에 종목 추가
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

# 비활성화
launchctl unload ~/Library/LaunchAgents/com.quant.trader.plist
```

---

## 모의 → 실투자 전환

`.env`에서:
```
KIS_MOCK=false
```

> 실투자 전환 전 반드시 모의투자로 충분히 검증하세요.

---

## 주의사항

1. `.env` 파일과 `stocks.py`는 절대 GitHub에 커밋하지 마세요.
2. API 키 없이 실행 시 자동으로 시뮬레이션 모드로 동작합니다.
3. 알고리즘 트레이딩은 예상치 못한 손실이 발생할 수 있습니다.

---

## 라이선스

MIT License — 개인 교육 및 연구 목적으로만 사용하세요.
