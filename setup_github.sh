#!/bin/bash
# setup_github.sh - GitHub 저장소 초기 설정 스크립트

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "  GitHub 저장소 설정"
echo "========================================="

# ── 1. git 설치 확인 ──
if ! command -v git &>/dev/null; then
    echo "  ✗ git이 설치되어 있지 않습니다."
    echo "    Homebrew로 설치: brew install git"
    exit 1
fi
echo "  ✓ git 버전: $(git --version)"

# ── 2. GitHub 저장소 주소 입력 ──
echo ""
read -rp "  GitHub 저장소 URL을 입력하세요 (예: https://github.com/username/quant_trader.git): " REPO_URL
if [ -z "$REPO_URL" ]; then
    echo "  ✗ URL이 입력되지 않았습니다. 종료합니다."
    exit 1
fi

# ── 3. git init ──
if [ ! -d ".git" ]; then
    git init
    echo "  ✓ git 저장소 초기화 완료"
else
    echo "  ⚠ 이미 git 저장소입니다."
fi

# ── 4. remote 설정 ──
if git remote | grep -q "^origin$"; then
    git remote set-url origin "$REPO_URL"
    echo "  ✓ origin remote URL 업데이트: $REPO_URL"
else
    git remote add origin "$REPO_URL"
    echo "  ✓ origin remote 추가: $REPO_URL"
fi

# ── 5. .env가 .gitignore에 포함되어 있는지 확인 ──
if grep -q "^\.env$" .gitignore; then
    echo "  ✓ .env가 .gitignore에 포함되어 있습니다. (안전)"
else
    echo "  ⚠ 경고: .env가 .gitignore에 없습니다! 절대 커밋하지 마세요."
fi

# ── 6. 스테이징 (.env 제외) ──
git add .
# .env가 실수로 스테이징된 경우 제거
git rm --cached .env 2>/dev/null || true
echo "  ✓ 파일 스테이징 완료 (.env 제외)"

# ── 7. 커밋 ──
git commit -m "init: 골든크로스 RSI 퀀트 전략 초기 세팅" 2>/dev/null || {
    echo "  ⚠ 커밋할 변경사항이 없거나 이미 커밋되었습니다."
}

# ── 8. main 브랜치로 설정 후 푸시 ──
git branch -M main
git push -u origin main

echo ""
echo "  ✓ GitHub 푸시 완료: $REPO_URL"
echo "========================================="
