#!/bin/bash
# readme-update.sh — .py 수정 시 Claude가 README.md 자동 업데이트
#
# 동작:
#   1. .py 파일 변경만 처리 (README 수정 시 재귀 방지)
#   2. 락으로 동시 실행 방지
#   3. claude CLI로 diff 기반 README 업데이트

LOCKFILE=/tmp/quant_readme_update.lock
REPO="$(cd "$(dirname "$0")/.." && pwd)"

read -r f
[[ "$f" =~ \.py$ ]] || exit 0
cd "$REPO" || exit 1

if ! mkdir "$LOCKFILE" 2>/dev/null; then
    exit 0
fi
trap 'rmdir "$LOCKFILE" 2>/dev/null; true' EXIT

sleep 2

DIFF=$(git diff HEAD -- "$f" 2>/dev/null | head -100)
BASENAME=$(basename "$f")

claude --dangerously-skip-permissions -p \
  "quant_trader 프로젝트에서 '${BASENAME}' 파일이 수정됐어.

변경 내용 (git diff):
${DIFF}

README.md를 읽고, 위 변경으로 인해 업데이트가 필요한 부분만 수정해줘.
- 새 기능/클래스/함수가 추가됐으면 반영
- 제거된 항목은 삭제
- 변경 없는 섹션은 건드리지 마
- README가 이미 최신이면 아무것도 하지 마" \
  --allowedTools "Read,Edit,Write" \
  2>/dev/null || true
