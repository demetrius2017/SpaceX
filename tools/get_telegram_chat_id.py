#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility: print chat_id from recent bot updates.
Usage:
  source .env
  python tools/get_telegram_chat_id.py
Then send a test message to your bot in Telegram and rerun if needed.
"""
import os
import sys
import json
import requests

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set", file=sys.stderr)
        sys.exit(1)
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"getUpdates failed: {e}", file=sys.stderr)
        sys.exit(2)
    data = r.json()
    print(json.dumps(data, ensure_ascii=False, indent=2))
    # Try to extract chat ids
    chats = set()
    for upd in data.get("result", []):
        msg = upd.get("message") or upd.get("channel_post") or {}
        chat = msg.get("chat") or {}
        if "id" in chat:
            chats.add((chat.get("id"), chat.get("type"), chat.get("title") or chat.get("username") or chat.get("first_name")))
    if chats:
        print("\nDetected chat ids:")
        for cid, ctype, name in chats:
            print(f" - {cid} ({ctype}) {name}")
    else:
        print("No chat ids found. Send any message to your bot and rerun.")

if __name__ == "__main__":
    main()
