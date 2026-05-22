#!/bin/bash
# git-autopush.sh — 연속 편집 시에도 안정적으로 커밋/푸시
#
# 동작 방식:
#   1. mkdir 으로 원자적 락 획득 (실패 시 pending 표시 후 종료)
#   2. 락 보유자가 2초 대기하며 연속 편집 묶음 처리
#   3. pending 플래그가 있는 한 계속 대기 (모든 편집 반영 후 커밋)
#   4. 변경된 파일 전체를 하나의 커밋으로 처리

LOCKDIR=/tmp/quant_autopush.lock
PENDING=/tmp/quant_autopush.pending
REPO=/Users/gyuyeong/quant_trader

read -r f
[[ "$f" =~ \.(py|md)$ ]] || exit 0
cd "$REPO" || exit 1

# 락 획득 시도 (mkdir은 atomic — 경쟁 조건 없음)
if ! mkdir "$LOCKDIR" 2>/dev/null; then
    # 다른 인스턴스 실행 중 → pending 표시 후 종료
    # 실행 중인 인스턴스가 pending을 감지해 처리
    touch "$PENDING"
    exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null; true' EXIT

# pending이 계속 들어오는 한 대기 (연속 편집 전부 묶기)
while true; do
    sleep 2
    if [ -f "$PENDING" ]; then
        rm -f "$PENDING"
        # pending 제거 후 추가 편집이 올 수 있으므로 한 번 더 대기
        continue
    fi
    break
done

git add -u
git diff --cached --quiet && exit 0

changed=$(git diff --cached --name-only | tr '\n' ' ' | sed 's/[[:space:]]*$//')
git commit -m "auto: ${changed} 수정"
git push origin main
