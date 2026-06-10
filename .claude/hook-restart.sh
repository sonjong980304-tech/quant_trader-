#!/bin/bash
# hook-restart.sh — .py 파일 수정 시 봇+스케줄러 재시작
LOG=/tmp/quant_hook_restart.log
f=$(python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    ti = d.get('tool_input') or {}
    tr = d.get('tool_response') or {}
    print(ti.get('file_path') or tr.get('filePath') or '')
except:
    print('')
" 2>>"$LOG")
echo "[$(date)] hook-restart: file=$f" >> "$LOG"
if echo "$f" | grep -q '\.py$'; then
    launchctl unload ~/Library/LaunchAgents/com.quant.telegrambot.plist >> "$LOG" 2>&1
    launchctl load  ~/Library/LaunchAgents/com.quant.telegrambot.plist >> "$LOG" 2>&1
    launchctl unload ~/Library/LaunchAgents/com.quant.trader.plist >> "$LOG" 2>&1
    launchctl load  ~/Library/LaunchAgents/com.quant.trader.plist >> "$LOG" 2>&1
    echo "[$(date)] hook-restart: 봇 재시작 완료" >> "$LOG"
fi
