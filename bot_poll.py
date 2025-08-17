#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram bot (long polling) для Heroku worker:
- Реагирует на /start: регистрирует подписчика, отправляет последнее письмо или свежую сводку.
- Использует ту же БД (db.py) и логику построения сводки, что и монитор.
"""
import os
import time
import json
from typing import Optional
import requests

from db import get_db
from monitor import build_telegram_text, load_yaml, build_summary, Event

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API = f"https://api.telegram.org/bot{TOKEN}" if TOKEN else None


def tg_api(method: str, payload: dict) -> dict:
    assert API, "TELEGRAM_BOT_TOKEN не задан"
    url = f"{API}/{method}"
    r = requests.post(url, json=payload, timeout=60)
    try:
        return r.json()
    except Exception:
        return {"ok": False, "status": r.status_code, "text": r.text[:200]}


def get_or_build_summary() -> Optional[dict]:
    try:
        cfg_path = "config.yml"
        checklist_path = "checklist.yaml"
        days_env = os.getenv("BROADCAST_DAYS")
        days = int(days_env) if days_env else None
        events, risk, growth, _ck, _cfg = build_summary(
            cfg_path=cfg_path,
            checklist_path=checklist_path,
            days=days,
            force_gpt=True,
        )
        return {
            "risk": risk,
            "growth": growth,
            "total": risk + growth,
            "events": [e.__dict__ for e in events],
        }
    except Exception:
        return None


def send_message(chat_id: int, text: str) -> None:
    tg_api("sendMessage", {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": False})


def handle_start(chat_id: int, username: Optional[str]) -> None:
    db = get_db()
    db.add_subscriber(chat_id, username)
    latest = db.get_latest_message()
    if latest:
        send_message(chat_id, latest)
        return
    summary = get_or_build_summary()
    if not summary:
        send_message(chat_id, "Сводка временно недоступна, попробуйте позже.")
        return
    cfg = load_yaml("config.yml")
    events = summary.get("events", [])
    risk = int(summary.get("risk", 0))
    growth = int(summary.get("growth", 0))
    events_obj = [Event(**e) for e in events]
    text = build_telegram_text(cfg, events_obj, risk, growth)
    send_message(chat_id, text)
    db.save_score(summary.get("total", 0), risk, growth, text)


def poll_loop() -> None:
    assert TOKEN, "TELEGRAM_BOT_TOKEN не задан"
    offset = None
    while True:
        try:
            payload = {"timeout": 50}
            if offset is not None:
                payload["offset"] = offset
            resp = tg_api("getUpdates", payload)
            if not resp.get("ok"):
                time.sleep(3)
                continue
            for upd in resp.get("result", []):
                offset = max(offset or 0, int(upd.get("update_id", 0)) + 1)
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                chat = msg.get("chat", {})
                chat_id = int(chat.get("id"))
                username = msg.get("from", {}).get("username")
                text = (msg.get("text") or "").strip()
                if text.startswith("/start"):
                    handle_start(chat_id, username)
        except Exception:
            time.sleep(5)


if __name__ == "__main__":
    poll_loop()
