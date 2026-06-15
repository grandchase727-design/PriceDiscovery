# -*- coding: utf-8 -*-
"""email_sender.py — Gmail SMTP sender with PDF attachment.

Reads credentials from .email_config.json (gitignored).
"""
from __future__ import annotations

import smtplib
import json
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication


CONFIG_PATH = Path(".email_config.json")


def load_config() -> dict:
    """Load email config from .email_config.json."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{CONFIG_PATH} not found. Create it with:\n"
            '{\n'
            '  "smtp_host": "smtp.gmail.com",\n'
            '  "smtp_port": 587,\n'
            '  "from_email": "grandchase727@gmail.com",\n'
            '  "from_password": "16char_app_password",\n'
            '  "to_email": "grandchase727@gmail.com"\n'
            "}"
        )
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def send_email_with_pdf(pdf_path: str, subject: str, body: str,
                        config: dict | None = None) -> dict:
    """Send email with PDF attachment via Gmail SMTP.

    Args:
        pdf_path: path to PDF file to attach
        subject: email subject
        body: HTML or plain text body
        config: email config (loads from file if None)

    Returns: status dict
    """
    if config is None:
        config = load_config()

    msg = MIMEMultipart()
    msg["From"] = config["from_email"]
    msg["To"] = config["to_email"]
    msg["Subject"] = subject

    # Body (plain text)
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # PDF attachment
    if Path(pdf_path).exists():
        with open(pdf_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=Path(pdf_path).name)
        part["Content-Disposition"] = f'attachment; filename="{Path(pdf_path).name}"'
        msg.attach(part)
    else:
        return {"ok": False, "error": f"PDF not found: {pdf_path}"}

    # SMTP send
    try:
        with smtplib.SMTP(config["smtp_host"], config["smtp_port"]) as server:
            server.starttls()
            server.login(config["from_email"], config["from_password"])
            server.send_message(msg)
        return {"ok": True, "to": config["to_email"], "subject": subject,
                "attachment": Path(pdf_path).name}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


if __name__ == "__main__":
    # Quick send test
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "/tmp/price_discovery_daily_test.pdf"
    result = send_email_with_pdf(
        pdf_path=pdf,
        subject="Price Discovery Daily Report — TEST",
        body=("Price Discovery 시스템 테스트 이메일입니다.\n\n"
              "PDF 첨부: 매수/매도 Final List 일일 리포트\n\n"
              "이 메시지가 수신되면 자동화 시스템이 정상 작동하고 있는 것입니다."),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
