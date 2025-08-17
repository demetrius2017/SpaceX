#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Send a test Telegram message using TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from env.
Usage:
  source .env
  python tools/send_telegram_test.py "Optional test text"
"""
import os
import sys
import requests

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set", file=sys.stderr)
        sys.exit(1)
    text = sys.argv[1] if len(sys.argv) > 1 else "✅ SpaceX Monitor: test message"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        r = requests.post(url, data=payload, timeout=10)
        print(f"Status: {r.status_code}")
        try:
            print(r.json())
        except Exception:
            print(r.text)
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
