#!/bin/bash
# hook-push.sh — PostToolUse에서 호출, python3으로 파일경로 추출 후 git-autopush 실행
LOG=/tmp/quant_hook_push.log
f=$(python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    ti = d.get('tool_input') or {}
    tr = d.get('tool_response') or {}
    print(ti.get('file_path') or tr.get('filePath') or '')
except Exception as e:
    import sys; print('', file=sys.stderr)
" 2>>"$LOG")
echo "[$(date)] hook-push: file=$f" >> "$LOG"
echo "$f" | bash /Users/gyuyeong/projects/quant_trader/.claude/git-autopush.sh >> "$LOG" 2>&1
