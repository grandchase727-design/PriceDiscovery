#!/bin/bash
# install_daily.sh — Install daily pipeline launchd job
#
# Schedules: Daily 07:00 KST execution of daily_pipeline.py
# Logs:      /tmp/daily_pipeline_<date>.log

set -e

PROJECT="/Users/parrot/Desktop/price discovery"
PLIST_SRC="$PROJECT/scripts/com.pricediscovery.daily.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.pricediscovery.daily.plist"
TG_CONFIG="$PROJECT/.telegram_config.json"

echo "============================================================"
echo "Price Discovery Daily Pipeline — Installer"
echo "============================================================"

# 1. Telegram config check
if [ ! -f "$TG_CONFIG" ]; then
    echo ""
    echo "⚠ .telegram_config.json 파일이 없습니다."
    echo ""
    echo "다음 단계로 설정하세요:"
    echo ""
    echo "[ 1 ] Telegram Bot 생성"
    echo "  • Telegram 앱에서 @BotFather 검색"
    echo "  • /newbot 명령어 → bot 이름 입력"
    echo "  • BotFather가 BOT_TOKEN 제공 (예: 1234567890:ABC-DEF...)"
    echo ""
    echo "[ 2 ] Chat ID 확인"
    echo "  • 생성한 bot에 아무 메시지 1번 전송"
    echo "  • 브라우저에서 접속:"
    echo "    https://api.telegram.org/bot<BOT_TOKEN>/getUpdates"
    echo "  • 'message.chat.id' 값 (숫자) 찾아서 복사"
    echo ""
    echo "[ 3 ] .telegram_config.json 작성"
    cat << 'EOF'
   {
     "bot_token": "1234567890:ABC-DEF...your_token",
     "chat_id":   "123456789"
   }
EOF
    echo ""
    echo "[ 4 ] 다시 이 스크립트 실행:  bash $0"
    echo ""
    exit 1
fi

echo "✓ .telegram_config.json 존재 확인"

# 2. Verify Python + dependencies
echo ""
echo "▸ Python 의존성 확인..."
python3 -c "import reportlab; print(f'  reportlab: {reportlab.__version__}')" || {
    echo "✗ reportlab 미설치. 설치: pip3 install reportlab"
    exit 1
}

# 3. Test Telegram connection
echo ""
echo "▸ Telegram bot 연결 테스트..."
cd "$PROJECT"
python3 -c "
from scripts.telegram_sender import send_message
r = send_message('🤖 <b>Price Discovery Daily Pipeline</b> 설치 테스트\n\n이 메시지가 보이면 Telegram 연동 성공!')
import json
print('  결과:', 'OK' if r.get('ok') else 'FAIL — ' + str(r.get('error','?'))[:100])
" || {
    echo "✗ Telegram 발송 실패 — bot_token + chat_id 확인 필요"
    exit 1
}

# 4. Check launchd dir
mkdir -p "$HOME/Library/LaunchAgents"

# 5. Unload existing (if any)
if launchctl list | grep -q "com.pricediscovery.daily"; then
    echo "▸ 기존 launchd job 제거..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# 6. Copy plist
echo "▸ launchd plist 설치..."
cp "$PLIST_SRC" "$PLIST_DST"
chmod 644 "$PLIST_DST"

# 7. Load
echo "▸ launchd job 등록..."
launchctl load "$PLIST_DST"

# 8. Verify
if launchctl list | grep -q "com.pricediscovery.daily"; then
    echo ""
    echo "============================================================"
    echo "✓ 설치 완료"
    echo "============================================================"
    echo ""
    echo "스케줄: 매일 07:00 KST (Mac 시간대 = Asia/Seoul 가정)"
    echo "Plist : $PLIST_DST"
    echo ""
    echo "확인 명령:"
    echo "  launchctl list | grep pricediscovery"
    echo ""
    echo "수동 trigger 테스트:"
    echo "  launchctl start com.pricediscovery.daily"
    echo ""
    echo "수동 실행 (스케줄 무관):"
    echo "  python3 \"$PROJECT/scripts/daily_pipeline.py\""
    echo ""
    echo "로그 위치:"
    echo "  stdout: /tmp/daily_pipeline_stdout.log"
    echo "  stderr: /tmp/daily_pipeline_stderr.log"
    echo "  날짜별: /tmp/daily_pipeline_<YYYY-MM-DD>.log"
    echo ""
    echo "제거 명령 (필요 시):"
    echo "  launchctl unload $PLIST_DST"
    echo "  rm $PLIST_DST"
    echo ""
else
    echo "✗ launchd job 등록 실패"
    exit 1
fi
