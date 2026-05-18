# 📈 퀀트 자동매매 시스템 — 골든크로스 + RSI 모멘텀 전략

> **투자 책임 고지**: 이 프로그램은 교육 및 연구 목적으로 제작되었습니다.  
> 실제 투자 손익에 대한 책임은 전적으로 사용자 본인에게 있습니다.  
> 과거 성과가 미래 수익을 보장하지 않습니다.

---

## 전략 소개

### 매수/매도 조건

| 구분 | 조건 |
|------|------|
| **매수** | 5일 이평선이 20일 이평선 **상향 돌파** (골든크로스) AND RSI(14) **≥ 55** |
| **전량 매도** | 5일 이평선이 20일 이평선 **하향 돌파** (데드크로스) |
| **분할 매도 50%** | RSI **75 이상** 도달 후 **70 밑으로** 하락 |

### 전략 파라미터 비교

| 항목 | 전략 A (기본) | 전략 B (비교) |
|------|--------------|--------------|
| 단기 이평선 | 5일 | 20일 |
| 장기 이평선 | 20일 | 60일 |
| RSI 매수 임계값 | 55 | 50 |

---

## 전체 아키텍처

```
[ macOS launchd (09:00 자동실행) ]
            │
            ▼
    [ graph.py - LangGraph 에이전트 ]
            │
    ┌───────┴───────────────────────────────────┐
    │                                           │
    ▼                                           │
[노드1] 데이터 수집 (yfinance)                  │
    │                                           │
    ▼                                           │
[노드2] 지표 계산 (MA5/20, RSI14)               │
    │                                           │
    ▼                                           │
[노드3] 신호 감지 ──── 신호 없음 ──────────► END
    │                                           │
    ▼                                           │
[노드4] 뉴스 수집 (Tavily API)                  │
    │                                           │
    ▼                                           │
[노드5] AI 판단 (GPT-4o) ─── 보류 ─────────► END
    │                                           │
    ▼                                           │
[노드6] 리스크 체크 (중복/연속 신호 필터)        │
    │                                           │
    ▼                                           │
[노드7] 알림 전송 (Telegram) + KIS API 주문      │
    │                                           │
    ▼                                           │
[노드8] 로그 저장 (logs/trader.log)             │
    │                                           │
    ▼                                           │
   END ◄──────────────────────────────────────┘

지원 종목: 삼성전자, SK하이닉스, NAVER, 카카오
```

---

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/YOUR_USERNAME/quant_trader.git
cd quant_trader
```

### 2. 자동 설치

```bash
bash install.sh
```

설치 스크립트가 다음을 자동으로 처리합니다:
- Python 패키지 설치
- `.env` 파일 생성 및 API 키 입력
- macOS launchd 자동실행 등록 (매일 09:00)

### 3. 수동 설치 (선택)

```bash
pip install -r requirements.txt
cp .env.example .env
# .env 파일을 편집기로 열어 API 키 입력
```

---

## API 키 발급 방법

### 텔레그램 봇 발급

1. Telegram에서 `@BotFather` 검색
2. `/newbot` 명령 입력 → 봇 이름 설정
3. 발급된 **Bot Token**을 `.env`의 `TELEGRAM_BOT_TOKEN`에 입력
4. 봇과 대화를 시작한 뒤 `https://api.telegram.org/bot<TOKEN>/getUpdates` 접속
5. `result[0].message.chat.id` 값을 `TELEGRAM_CHAT_ID`에 입력

### KIS API 키 발급

1. [한국투자증권 KIS Developers](https://apiportal.koreainvestment.com) 접속
2. 회원가입 및 로그인
3. `Apps → 나의 애플리케이션 → 서비스 신청`
4. 모의투자 또는 실투자 선택 후 신청
5. 발급된 **App Key**와 **App Secret**을 `.env`에 입력

### OpenAI API 키 발급

1. [OpenAI Platform](https://platform.openai.com) 접속
2. `API Keys → Create new secret key`
3. 발급된 키를 `.env`의 `OPENAI_API_KEY`에 입력

### Tavily API 키 발급

1. [Tavily](https://tavily.com) 접속 → 회원가입
2. 대시보드에서 API Key 확인
3. `.env`의 `TAVILY_API_KEY`에 입력

---

## 모의투자 → 실투자 전환

`config.py` 또는 `.env` 파일 수정:

```bash
# .env 파일에서
KIS_MOCK=false
```

또는 `config.py`에서:

```python
# IS_MOCK = True 이면 모의투자, False이면 실투자
IS_MOCK = False  # 실투자 전환
```

> ⚠️ **주의**: 실투자 전환 전 반드시 모의투자로 충분히 테스트하세요.

---

## 백테스트 실행

```bash
python backtest.py
```

출력 항목: CAGR, MDD, Sharpe Ratio, 승률, 총 매매횟수  
차트 저장: `results/backtest_result.png`

---

## 자동매매 수동 실행

```bash
python graph.py
```

---

## macOS 자동실행 관리

```bash
# 자동실행 활성화 (install.sh에서 자동 처리)
launchctl load ~/Library/LaunchAgents/com.quant.trader.plist

# 자동실행 비활성화
launchctl unload ~/Library/LaunchAgents/com.quant.trader.plist

# 즉시 실행 테스트
launchctl start com.quant.trader
```

### 절전모드 방지 설정

Mac Mini를 항상 켜두려면:

```bash
# 전원 연결 시 절전모드 비활성화
sudo pmset -c sleep 0
sudo pmset -c disksleep 0

# 설정 확인
pmset -g
```

또는: **시스템 설정 → 배터리 → 전원 어댑터 → "잠자기 방지"** 활성화

---

## 프로젝트 구조

```
quant_trader/
├── .env                  # API 키 (절대 공개 금지)
├── .env.example          # 키 목록 템플릿
├── .gitignore
├── config.py             # 전략 파라미터 및 환경설정
├── data_fetcher.py       # yfinance 데이터 수집
├── indicators.py         # 기술적 지표 계산
├── strategy.py           # 매수/매도 신호 생성
├── backtest.py           # vectorbt 백테스트
├── notifier.py           # 텔레그램 알림
├── trader.py             # KIS API 주문 실행
├── graph.py              # LangGraph 에이전트
├── requirements.txt
├── install.sh            # 자동 설치 스크립트
├── setup_github.sh       # GitHub 초기 설정
├── com.quant.trader.plist  # macOS launchd 설정
├── logs/
│   └── trader.log        # 매매 로그
└── results/
    └── backtest_result.png  # 백테스트 차트
```

---

## 주의사항

1. **API 키 보안**: `.env` 파일은 절대 GitHub에 커밋하지 마세요.
2. **모의투자 우선**: 실투자 전환 전 최소 1개월 모의투자 검증을 권장합니다.
3. **시장 개장 시간**: 한국 주식 시장은 09:00~15:30 (평일)입니다.
4. **네트워크 의존성**: 인터넷 연결이 필요합니다.
5. **투자 손실 위험**: 알고리즘 트레이딩은 예상치 못한 손실이 발생할 수 있습니다.

---

## 라이선스

MIT License — 개인 교육 및 연구 목적으로만 사용하세요.
