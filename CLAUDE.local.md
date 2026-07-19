# quant_trader 프로젝트 하네스

## 프로젝트 개요
퀀트 트레이딩 봇 프로젝트. 텔레그램 봇으로 알림/제어를 처리한다.

## 자동 훅 (`.claude/settings.json`)
- `.py` / `.md` 파일 수정 시 → **GitHub 자동 커밋·푸시** (`.claude/hook-push.sh` → `.claude/git-autopush.sh`)
- `.py` 파일 수정 시 → 텔레그램 봇 + 스케줄러 자동 재시작 (`com.quant.telegrambot.plist` + `com.quant.trader.plist`)

> ⚠️ **README.md 자동 업데이트는 현재 비활성.** `.claude/readme-update.sh` 스크립트 파일은 존재하지만 어떤 훅에도 등록돼 있지 않아 실제로는 동작하지 않는다. (2026-07-02 확인)

## 훅 사용 규칙 (반드시 준수)
1. **수동 git commit / git push 금지** — 훅이 자동 처리하므로 직접 실행하지 않는다
2. 봇 재시작은 수동으로 할 필요 없음 — 훅이 자동 처리
3. **README.md는 수동으로 관리한다** — 자동 반영 기능은 현재 비활성이므로 직접 업데이트할 것

## 주요 파일
- `backtest/backtest_ml.py` — 최근 1개월 ML 백테스트
- `runner.py` — EOD 스케줄러 (15:31 KR EOD 신호 스캔, reversion+trend 슬롯분리 운용) *(루트 유지)*
- `telegram_bot.py` — 텔레그램 봇 *(루트 유지)*
- `interface/langchain_agent.py` — LangChain AI 어시스턴트
- `core/trader.py` — KIS API (국내)
- `com.quant.telegrambot.plist` — 텔레그램 봇 launchd 설정

> 2026-07 폴더 재구성: 루트의 매매/전략/데이터/백테스트 모듈을 역할별 패키지로 이동
> (`core/` 매매·주문, `strategy/` 신호·지표·레짐, `data/` 수집, `interface/` 알림·AI,
> `backtest/` 백테스트, `scripts/` 수동 실행). `config.py`·`runner.py`·`telegram_bot.py`는 루트 유지.

## 주의사항
- launchd plist는 `~/Library/LaunchAgents/`에 심볼릭 링크 또는 복사되어 있어야 함
