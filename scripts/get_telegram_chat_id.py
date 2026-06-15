#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""get_telegram_chat_id.py — Bot에 메시지 보낸 후 chat_id 자동 조회.

Usage:
  1) 먼저 Telegram에서 본인 bot에 '/start' 또는 아무 메시지 전송
  2) 이 스크립트 실행:
     python3 scripts/get_telegram_chat_id.py <BOT_TOKEN>

Example:
  python3 scripts/get_telegram_chat_id.py 1234567890:ABC-DEF...
"""
import sys
import json
import urllib.request


def get_chat_ids(bot_token: str) -> list:
    """Fetch getUpdates and extract chat_id list."""
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"✗ API 호출 실패: {e}")
        print(f"  Bot token이 정확한지 확인하세요.")
        return []

    if not data.get("ok"):
        print(f"✗ Telegram API 에러: {data}")
        return []

    results = data.get("result", []) or []
    if not results:
        print("⚠ Updates가 비어있습니다.")
        print()
        print("다음 단계로 진행하세요:")
        print("  1) Telegram 앱에서 본인 bot을 찾아 채팅 열기")
        print("  2) 아무 메시지 (예: /start) 전송")
        print("  3) 다시 이 스크립트 실행")
        return []

    chat_info = {}
    for upd in results:
        msg = upd.get("message") or upd.get("edited_message") or {}
        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None:
            continue
        chat_info[chat_id] = {
            "type": chat.get("type"),
            "username": chat.get("username") or "—",
            "first_name": chat.get("first_name") or "",
            "last_name": chat.get("last_name") or "",
            "title": chat.get("title") or "",
        }
    return list(chat_info.items())


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/get_telegram_chat_id.py <BOT_TOKEN>")
        print()
        print("Bot Token 받는 법:")
        print("  1) Telegram에서 @BotFather 검색")
        print("  2) /newbot → bot 이름 + username 지정")
        print("  3) BotFather가 token 발급 (예: 1234567890:ABC-DEF...)")
        sys.exit(1)

    bot_token = sys.argv[1].strip()
    if not bot_token or ":" not in bot_token:
        print(f"✗ Bot token 형식 오류: '{bot_token}'")
        print("  올바른 형식: 1234567890:ABC-DEF1234...")
        sys.exit(1)

    print("=" * 60)
    print(f"Bot token 확인 중...")
    print("=" * 60)
    # First check bot itself
    me_url = f"https://api.telegram.org/bot{bot_token}/getMe"
    try:
        with urllib.request.urlopen(me_url, timeout=10) as resp:
            me = json.loads(resp.read().decode())
        if me.get("ok"):
            bot = me["result"]
            print(f"✓ Bot 인증 성공:")
            print(f"  이름: {bot.get('first_name')}")
            print(f"  Username: @{bot.get('username')}")
            print(f"  Bot ID: {bot.get('id')}")
        else:
            print(f"✗ Bot 인증 실패: {me}")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Bot 인증 실패: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("Chat ID 조회 중...")
    print("=" * 60)

    chats = get_chat_ids(bot_token)
    if not chats:
        sys.exit(1)

    print(f"\n✓ {len(chats)}개의 chat 발견:\n")
    for chat_id, info in chats:
        marker = "👤" if info["type"] == "private" else "👥"
        name = f"{info['first_name']} {info['last_name']}".strip() or info["title"] or "—"
        print(f"  {marker} chat_id = {chat_id}")
        print(f"     이름: {name}")
        print(f"     Username: @{info['username']}")
        print(f"     Type: {info['type']}")
        print()

    if len(chats) == 1:
        cid = chats[0][0]
        print("=" * 60)
        print("✓ .telegram_config.json 자동 생성 가능!")
        print("=" * 60)
        print()
        print(f"이 명령어로 config 생성:")
        print()
        print(f'cat > .telegram_config.json << EOF')
        print(f'{{')
        print(f'  "bot_token": "{bot_token}",')
        print(f'  "chat_id": "{cid}"')
        print(f'}}')
        print(f'EOF')

    return 0


if __name__ == "__main__":
    sys.exit(main())
