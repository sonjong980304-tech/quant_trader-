#!/bin/bash
# install.sh - 퀀트 자동매매 시스템 자동 설치 스크립트

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "  퀀트 자동매매 시스템 설치 시작"
echo "========================================="

# ── 1. Python 패키지 설치 ──
echo ""
echo "[1/5] Python 패키지 설치 중..."
pip install -r requirements.txt
echo "  ✓ 패키지 설치 완료"

# ── 2. .env 파일 생성 ──
echo ""
echo "[2/5] 환경변수 파일 설정..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  ✓ .env 파일 생성 완료"
else
    echo "  ⚠ .env 파일이 이미 존재합니다. 덮어쓰지 않습니다."
fi

# ── 3. API 키 입력 받기 ──
echo ""
echo "[3/5] API 키 설정"
echo "  (Enter 키를 누르면 해당 항목은 건너뜁니다)"

read_and_set() {
    local key_name="$1"
    local prompt_msg="$2"
    printf "  %s: " "$prompt_msg"
    read -r value
    if [ -n "$value" ]; then
        # macOS sed는 -i '' 사용
        if grep -q "^${key_name}=" .env; then
            sed -i '' "s|^${key_name}=.*|${key_name}=${value}|" .env
        else
            echo "${key_name}=${value}" >> .env
        fi
        echo "    ✓ ${key_name} 저장됨"
    fi
}

read_and_set "KIS_APP_KEY"         "KIS App Key"
read_and_set "KIS_APP_SECRET"      "KIS App Secret"
read_and_set "TELEGRAM_BOT_TOKEN"  "Telegram Bot Token"
read_and_set "TELEGRAM_CHAT_ID"    "Telegram Chat ID"
read_and_set "OPENAI_API_KEY"      "OpenAI API Key"
read_and_set "TAVILY_API_KEY"      "Tavily API Key"

echo "  ✓ .env 설정 완료"

# ── 4. launchd plist 등록 (macOS 자동실행) ──
echo ""
echo "[4/5] macOS 자동실행 등록 중..."

PLIST_NAME="com.quant.trader"
PLIST_SRC="${SCRIPT_DIR}/com.quant.trader.plist"
PLIST_DST="${HOME}/Library/LaunchAgents/${PLIST_NAME}.plist"

# plist 파일 동적 생성 (실제 경로 삽입)
PYTHON_PATH="$(which python3)"
cat > "${PLIST_SRC}" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_PATH}</string>
        <string>${SCRIPT_DIR}/graph.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${SCRIPT_DIR}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${SCRIPT_DIR}/logs/trader.log</string>
    <key>StandardErrorPath</key>
    <string>${SCRIPT_DIR}/logs/trader.log</string>
    <key>RunAtLoad</key>
    <false/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
PLIST_EOF

# LaunchAgents 폴더 없으면 생성
mkdir -p "${HOME}/Library/LaunchAgents"
cp "${PLIST_SRC}" "${PLIST_DST}"

# 기존 plist 언로드 (오류 무시)
launchctl unload "${PLIST_DST}" 2>/dev/null || true
launchctl load "${PLIST_DST}"

echo "  ✓ launchd 등록 완료 → 매일 09:00 자동 실행"
echo "  ※ 절전모드 방지: 시스템 설정 → 배터리 → '전원 어댑터' → '잠자기 방지' 활성화"
echo "  ※ pmset 명령어 사용 시: sudo pmset -a sleep 0"

# ── 5. GitHub 설정 ──
echo ""
echo "[5/5] GitHub 설정"
read -rp "  GitHub 저장소를 지금 설정하시겠습니까? (y/N): " do_github
if [[ "$do_github" =~ ^[Yy]$ ]]; then
    bash setup_github.sh
fi

echo ""
echo "========================================="
echo "  설치 완료!"
echo "========================================="
echo ""
echo "  다음 명령어로 실행할 수 있습니다:"
echo "    python backtest.py   # 백테스트 실행"
echo "    python graph.py      # 자동매매 에이전트 실행"
echo ""
