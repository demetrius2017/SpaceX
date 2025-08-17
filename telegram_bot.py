#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилиты Telegram-бота для ежедневной рассылки (без вебхука).
Используется с worker (long polling) и Heroku Scheduler.
"""
import os
import json
from typing import Optional
import requests

from db import get_db
from monitor import build_telegram_text, load_yaml, build_summary, Event

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


def tg_send(chat_id: int, text: str) -> None:
    if not BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": False}, timeout=20)


def get_or_build_summary() -> Optional[dict]:
    try:
        cfg_path = "config.yml"
        checklist_path = "checklist.yaml"
        days_env = os.getenv("BROADCAST_DAYS")
        days = int(days_env) if days_env else None
        # Всегда строим одну и ту же сводку, предпочтительно GPT при наличии ключа
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


def daily_broadcast() -> None:
    """Отправить ежедневное обновление всем подписчикам с дельтой к вчерашнему total."""
    db = get_db()
    summary = get_or_build_summary()
    if not summary:
        return
    cfg = load_yaml("config.yml")
    total = int(summary.get("total", 0))
    risk = int(summary.get("risk", 0))
    growth = int(summary.get("growth", 0))
    prev = db.get_prev_score()
    delta_str = ""
    if prev:
        prev_total = prev[0]
        diff = total - prev_total
        sign = "+" if diff >= 0 else "−"
        delta_str = f" (Δ {sign}{abs(diff)})"

    events = summary.get("events", [])
    events_obj = [Event(**e) for e in events]
    msg = build_telegram_text(cfg, events_obj, risk, growth)
    msg = msg.replace("индекс:", f"индекс:{delta_str}")

    for chat_id in db.get_subscribers():
        tg_send(chat_id, msg)

    db.save_score(total, risk, growth, msg)


if __name__ == "__main__":
    # Однократный запуск ежедневной рассылки (для локальных тестов)
    daily_broadcast()
