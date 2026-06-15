# -*- coding: utf-8 -*-
"""telegram_sender.py — Telegram Bot API PDF sender.

Reads bot credentials from .telegram_config.json (gitignored).

Setup:
  1. Create Telegram bot via @BotFather → get TOKEN
  2. Send any message to your bot, then visit:
     https://api.telegram.org/bot<TOKEN>/getUpdates
     to find your chat_id (numeric)
  3. Create .telegram_config.json:
     {
       "bot_token": "1234567890:ABC-DEF...",
       "chat_id": "123456789"
     }
"""
from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Optional


CONFIG_PATH = Path(".telegram_config.json")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{CONFIG_PATH} not found. Create it with:\n"
            '{\n'
            '  "bot_token": "your_bot_token_from_botfather",\n'
            '  "chat_id":   "your_telegram_chat_id"\n'
            "}"
        )
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def send_message(text: str, config: Optional[dict] = None) -> dict:
    """Send a text message via Telegram."""
    import urllib.request, urllib.parse
    if config is None:
        config = load_config()
    url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": config["chat_id"],
        "text": text,
        "parse_mode": "HTML",
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {"ok": True, "response": json.loads(resp.read().decode())}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def send_document(file_path: str, caption: str = "",
                   config: Optional[dict] = None) -> dict:
    """Send a file (PDF) via Telegram using multipart/form-data."""
    import urllib.request
    from email.mime.multipart import MIMEMultipart
    from email.mime.application import MIMEApplication

    if config is None:
        config = load_config()

    p = Path(file_path)
    if not p.exists():
        return {"ok": False, "error": f"file not found: {file_path}"}

    url = f"https://api.telegram.org/bot{config['bot_token']}/sendDocument"

    # Manual multipart construction (no external deps)
    boundary = "----PriceDiscoveryBoundary7K2nF9xL3qZ"
    chat_id = str(config["chat_id"])
    mime_type, _ = mimetypes.guess_type(str(p))
    mime_type = mime_type or "application/octet-stream"

    body_parts: list[bytes] = []
    # chat_id
    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(b'Content-Disposition: form-data; name="chat_id"\r\n\r\n')
    body_parts.append(chat_id.encode() + b"\r\n")
    # caption
    if caption:
        body_parts.append(f"--{boundary}\r\n".encode())
        body_parts.append(b'Content-Disposition: form-data; name="caption"\r\n\r\n')
        body_parts.append(caption.encode("utf-8") + b"\r\n")
    # parse_mode
    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(b'Content-Disposition: form-data; name="parse_mode"\r\n\r\n')
    body_parts.append(b"HTML\r\n")
    # document file
    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(
        f'Content-Disposition: form-data; name="document"; filename="{p.name}"\r\n'.encode()
    )
    body_parts.append(f"Content-Type: {mime_type}\r\n\r\n".encode())
    body_parts.append(p.read_bytes())
    body_parts.append(b"\r\n")
    body_parts.append(f"--{boundary}--\r\n".encode())

    body = b"".join(body_parts)

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return {"ok": True, "size_kb": p.stat().st_size / 1024,
                    "filename": p.name, "chat_id": chat_id}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


def send_pdf_with_summary(pdf_path: str, summary: dict,
                           config: Optional[dict] = None) -> dict:
    """Send PDF with a Korean summary caption.

    Args:
        pdf_path: PDF file path
        summary: dict with date, n_buy, n_sell, etc.
        config: optional Telegram config
    """
    date_str = summary.get("date", "")
    n_buy = summary.get("n_buy", 0)
    n_sell = summary.get("n_sell", 0)
    top_buy = summary.get("top_buy_tickers", "")
    top_sell = summary.get("top_sell_tickers", "")

    caption = (
        f"📊 <b>Price Discovery Daily Report</b>\n"
        f"📅 {date_str}\n\n"
        f"🟢 매수 Final List: <b>{n_buy}개</b>\n"
        f"🔴 매도 Final List: <b>{n_sell}개</b>\n\n"
    )
    if top_buy:
        caption += f"🌟 Top 매수: {top_buy}\n"
    if top_sell:
        caption += f"⚠ Top 매도: {top_sell}\n"
    caption += "\n3-Agent Voting (PM + Trading + Risk) 기반 자동 분석\n⚠ 투자 자문 아님"

    # Telegram caption max = 1024 chars
    caption = caption[:1020]

    return send_document(pdf_path, caption=caption, config=config)


if __name__ == "__main__":
    import sys
    # Quick send test
    pdf = sys.argv[1] if len(sys.argv) > 1 else "/tmp/price_discovery_daily_test.pdf"
    if not Path(pdf).exists():
        print(f"PDF not found at {pdf} — run pdf_generator.py first")
        sys.exit(1)
    result = send_pdf_with_summary(
        pdf_path=pdf,
        summary={
            "date": "TEST",
            "n_buy": 47,
            "n_sell": 45,
            "top_buy_tickers": "IWD, MGV, SPHQ",
            "top_sell_tickers": "ZS, PLTR, MSCI",
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
