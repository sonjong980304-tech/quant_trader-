# quant_trader 프로젝트 하네스

## 프로젝트 개요
퀀트 트레이딩 봇 프로젝트. 텔레그램 봇으로 알림/제어를 처리한다.

## 자동 훅 (`.claude/settings.json`)
- `.py` / `.md` 파일 수정 시 → **GitHub 자동 커밋·푸시** (`.claude/git-autopush.sh`)
- `.py` 파일 수정 시 → 텔레그램 봇 자동 재시작 (`com.quant.telegrambot.plist`)
- `.py` 파일 수정 시 → **README.md 자동 업데이트** (`.claude/readme-update.sh`) — git diff 기반으로 Claude가 변경 내용만 반영

## 훅 사용 규칙 (반드시 준수)
1. **수동 git commit / git push 금지** — 훅이 자동 처리하므로 직접 실행하지 않는다
2. README.md 수동 업데이트 불필요 — `.py` 수정 시 훅이 자동 반영
3. 봇 재시작도 수동으로 할 필요 없음 — 훅이 자동 처리

## 주요 파일
- `backtest_ml.py` — 최근 1개월 ML 백테스트
- `runner.py` — 장중 스케줄러 (MA/RSI + 급등주 스캔 + 리밸런싱)
- `telegram_bot.py` — 텔레그램 봇
- `langchain_agent.py` — LangChain AI 어시스턴트
- `trader.py` — KIS API (국내 + 미국주식)
- `com.quant.telegrambot.plist` — 텔레그램 봇 launchd 설정
- `com.quant.dashboard.plist` — 대시보드 launchd 설정

## 주의사항
- launchd plist는 `~/Library/LaunchAgents/`에 심볼릭 링크 또는 복사되어 있어야 함
