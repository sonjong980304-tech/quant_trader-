#!/bin/bash
# hook-restart.sh — .py 파일 수정 시 봇+스케줄러 재시작
LOG=/tmp/quant_hook_restart.log

echo "[$(date)] hook-restart: 훅 호출됨" >> "$LOG"

# stdin을 변수에 먼저 캡처 (스트림은 한 번만 읽을 수 있음)
raw=$(cat)
echo "[$(date)] hook-restart: stdin_preview=$(echo "$raw" | head -c 300 | tr '\n' ' ')" >> "$LOG"

f=$(echo "$raw" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    ti = d.get('tool_input') or {}
    tr = d.get('tool_response') or {}
    print(ti.get('file_path') or tr.get('filePath') or '')
except Exception as e:
    sys.stderr.write(f'parse_error: {e}\n')
    print('')
" 2>>"$LOG")

echo "[$(date)] hook-restart: file=$f" >> "$LOG"

if echo "$f" | grep -q '\.py$'; then
    launchctl unload ~/Library/LaunchAgents/com.quant.telegrambot.plist >> "$LOG" 2>&1
    launchctl load  ~/Library/LaunchAgents/com.quant.telegrambot.plist >> "$LOG" 2>&1
    launchctl unload ~/Library/LaunchAgents/com.quant.trader.plist >> "$LOG" 2>&1
    launchctl load  ~/Library/LaunchAgents/com.quant.trader.plist >> "$LOG" 2>&1
    echo "[$(date)] hook-restart: 봇 재시작 완료" >> "$LOG"
else
    echo "[$(date)] hook-restart: .py 아님, 재시작 스킵" >> "$LOG"
fi
