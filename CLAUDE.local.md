# quant_trader 프로젝트 하네스

## 프로젝트 개요
퀀트 트레이딩 봇 프로젝트. 텔레그램 봇으로 알림/제어를 처리한다.

## 자동 훅
- `.py` 파일 수정 시 → 텔레그램 봇 자동 재시작 (`com.quant.telegrambot.plist`)
- 설정 위치: `.claude/settings.json`

## 주요 파일
- `backtest.py` — 백테스트 실행
- `com.quant.telegrambot.plist` — 텔레그램 봇 launchd 설정
- `com.quant.dashboard.plist` — 대시보드 launchd 설정

## 주의사항
- Python 파일 수정 후 봇이 자동 재시작되므로 별도로 재시작할 필요 없음
- launchd plist는 `~/Library/LaunchAgents/`에 심볼릭 링크 또는 복사되어 있어야 함
