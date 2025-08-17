#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpaceX GPT Monitor
==================

Ежедневно оценивает тональность новостей о SpaceX, используя OpenAI API,
и при значимом изменении отправляет уведомление в Telegram.

Требования: pip install -r requirements.txt (нужен openai>=1.40)
Переменные окружения:
  OPENAI_API_KEY       – API‑ключ OpenAI
  TELEGRAM_BOT_TOKEN   – токен бота Telegram (BotFather)
  TELEGRAM_CHAT_ID     – chat id или @username для отправки отчётов
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import feedparser
import requests
import yaml

# OpenAI SDK v1.x
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36; spacex-monitor/1.0"
}

def load_sources(config_path: str = "config.yml") -> List[str]:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sources: List[str] = []
    for src in cfg.get("sources", []):
        if src.get("enabled", True) and src.get("url") and src.get("type") == "rss":
            sources.append(src["url"])
    return sources

def fetch_news(urls: Iterable[str]) -> List[Tuple[str, str, str]]:
    items: List[Tuple[str, str, str]] = []
    for url in urls:
        try:
            # Подменяем загрузку: сначала requests (с заголовками), затем feedparser.parse(bytes)
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            feed = feedparser.parse(r.content)
        except Exception:
            continue
        for entry in getattr(feed, "entries", []) or []:
            title = (getattr(entry, "title", "") or "").strip()
            link = (getattr(entry, "link", "") or "").strip()
            date = ((getattr(entry, "published", None) or getattr(entry, "updated", None)) or "").strip()
            if title:
                items.append((title, link, date))
    return items

SYSTEM_PROMPT = (
    "You are a financial news sentiment classifier. Given a news headline about SpaceX, "
    "respond with a single word indicating the overall sentiment for SpaceX: 'positive', "
    "'neutral', or 'negative'."
)

def classify_sentiment(headlines: Iterable[Tuple[str, str, str]], model: str = "gpt-4o-mini") -> List[Tuple[str, str, str, str]]:
    if OpenAI is None:
        raise RuntimeError("openai library is not installed. Add it to requirements and install.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)

    results: List[Tuple[str, str, str, str]] = []
    # Простая поштучная классификация (можно батчить для экономии)
    for title, link, date in headlines:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Headline: {title}"},
                ],
                max_tokens=3,
                temperature=0.0,
            )
            content = (resp.choices[0].message.content or "").strip().lower()
            if "positive" in content:
                sentiment = "positive"
            elif "negative" in content:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        except Exception:
            sentiment = "neutral"
        results.append((title, link, date, sentiment))
    return results

def compute_score(items: Iterable[Tuple[str, str, str, str]]) -> int:
    score_map = {"positive": 1, "neutral": 0, "negative": -1}
    return sum(score_map.get(sentiment, 0) for _, _, _, sentiment in items)

def load_previous_score(path: str = "prev_score.json") -> dict:
    if Path(path).exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_score(score: int, path: str = "prev_score.json") -> None:
    from datetime import datetime, timezone
    data = {"score": score, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def send_telegram(score: int, diff: int, items: List[Tuple[str, str, str, str]]) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Telegram credentials not set (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)")
    if score >= 10:
        recommendation = "Strong Buy"
    elif score >= 5:
        recommendation = "Buy"
    elif score >= 1:
        recommendation = "Hold"
    elif score >= -4:
        recommendation = "Trim"
    else:
        recommendation = "Reduce"
    sign = "+" if diff >= 0 else ""
    lines = [
        f"SpaceX sentiment score: {score} (change: {sign}{diff})",
        f"Recommendation: {recommendation}",
        "",
        "Recent headlines:",
    ]
    for title, link, date, sentiment in items[:10]:
        emoji = {"positive": "🟢", "neutral": "⚪️", "negative": "🔴"}.get(sentiment, "⚪️")
        lines.append(f"{emoji} {sentiment.title()}: {title}")
    message = "\n".join(lines)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception:
        pass

def main() -> None:
    sources = load_sources()
    if not sources:
        print("No sources enabled in config.yml", file=sys.stderr)
        sys.exit(1)
    headlines = fetch_news(sources)
    if not headlines:
        print("No headlines fetched.")
        sys.exit(0)
    classified = classify_sentiment(headlines)
    current_score = compute_score(classified)
    prev = load_previous_score()
    prev_score = prev.get("score", 0)
    diff = current_score - prev_score
    save_score(current_score)
    send_telegram(current_score, diff, classified)
    print(f"Score: {current_score} (delta {diff}) – sent Telegram notification.")

if __name__ == "__main__":
    main()
