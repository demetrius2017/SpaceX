#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpaceX Monitoring Dashboard
Автор: GPT-5 Pro (ведущий разработчик, отвечает за код)
Описание: собирает события из публичных источников (RSS/HTML),
применяет правила-ключевые слова, считает индекс риска/роста и формирует HTML-дашборд.
"""
import argparse
import logging
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Iterable
from urllib.parse import urljoin

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from jinja2 import Environment, FileSystemLoader, select_autoescape
from tenacity import retry, stop_after_attempt, wait_fixed
from rich.console import Console
from rich.table import Table
from rich import box
from typing import cast

try:
    # OpenAI SDK v1.x
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

console = Console()
logger = logging.getLogger("spacex_monitor")

# ------------------------------- Utilities ----------------------------------

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def norm_dt(dt: Any) -> Optional[datetime]:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt
    try:
        return dateparser.parse(str(dt))
    except Exception:
        return None

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def _title_fingerprint(title: str, max_words: int = 12) -> str:
    """Грубое нормализованное представление заголовка для дедупликации.
    - lower, убираем небуквенные/нецифровые символы
    - схлопываем пробелы
    - берём первые max_words слов
    """
    import re
    t = (title or "").lower()
    # убрать содержимое в круглых скобках, например (HLS)
    t = re.sub(r"\([^\)]*\)", " ", t)
    t = re.sub(r"[^a-z0-9а-яё\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    words = [w for w in t.split() if w not in {"hls"}]
    if max_words and len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)

def _valuation_key(title: str) -> Optional[str]:
    """Выделяет ключ оценки для финансовых новостей, напр. valuation:400.
    Ищем конструкции вида:
    - $400 billion / 400 billion / 400bn
    - 400 млрд / 400 миллиарда / 400 миллиардов
    Возвращаем строку вида 'valuation:400' или None.
    """
    import re
    t = (title or "").lower()
    # Нормализуем разделители
    t = t.replace("миллиардов", "миллиард").replace("миллиарда", "миллиард").replace("billion", "bn").replace("млрд", "миллиард")
    # Паттерны: 400 bn | 400 миллиард | $400 bn
    m = re.search(r"\$?\s*(\d{2,4})(?:[\s\u00a0]*[,\.]\d+)?\s*(bn|миллиард)", t)
    if not m:
        # Попробуем вариант: 'оценк' рядом с числом
        m = re.search(r"оценк\w*[^\d]{0,6}(\d{2,4})", t)
        if not m:
            return None
    try:
        num = int(m.group(1))
        # Ограничим разумными значениями
        if 10 <= num <= 2000:
            return f"valuation:{num}"
    except Exception:
        return None
    return None

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ----------------------------- Data Classes ---------------------------------

@dataclass
class Rule:
    name: str
    patterns: List[str]
    categories: List[str]
    weight: int
    tag: str
    impact: str  # "risk" | "growth"

@dataclass
class Event:
    id: str
    title: str
    url: str
    source: str
    published: Optional[str]
    category: str
    tag: str
    weight: int
    impact: str

# ----------------------------- Load Config ----------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_jsonl(path: str, records: Iterable[dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl_map(path: str) -> Dict[str, dict]:
    """Загружает JSONL в карту id -> объект."""
    data: Dict[str, dict] = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = obj.get("id") or obj.get("key")
                if isinstance(k, str):
                    data[k] = obj
            except Exception:
                continue
    return data

def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out

# ----------------------------- Fetchers -------------------------------------

HEADERS = {
    # Более реалистичный UA помогает некоторым сайтам отдавать контент
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36; spacex-monitor/1.0"
}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def get(url: str, timeout: int = 20) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp

def fetch_rss(url: str) -> List[dict]:
    logger.debug(f"fetch_rss: GET {url}")
    # Скачиваем фид сами (с заголовками и ретраями), затем парсим
    try:
        r = get(url)
        content = r.content
        d = feedparser.parse(content)
    except Exception as e:
        logger.debug(f"fetch_rss: error fetching {url}: {e}")
        d = feedparser.parse("")
    items = []
    for e in d.entries:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        summary = getattr(e, "summary", "")
        published = getattr(e, "published", None) or getattr(e, "updated", None)
        items.append({
            "title": title,
            "url": link,
            "summary": BeautifulSoup(summary, "html.parser").get_text(" ", strip=True) if summary else "",
            "published": norm_dt(published),
        })
    logger.debug(f"fetch_rss: parsed {len(items)} items from {url}")
    return items

def fetch_html_generic(url: str, css_item: str, css_title: str, css_url_selector: Optional[str], css_url_attr: str, css_date: Optional[str]) -> List[dict]:
    logger.debug(f"fetch_html_generic: GET {url}")
    r = get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    records = []
    for node in soup.select(css_item):
        # title
        tnode = node.select_one(css_title) if css_title else None
        title = tnode.get_text(" ", strip=True) if tnode else None
        if not title:
            continue
        # url
        link: Optional[object] = None
        if css_url_selector:
            unode = node.select_one(css_url_selector)
            if unode and unode.has_attr(css_url_attr):
                link = unode[css_url_attr]
        else:
            if tnode and tnode.has_attr(css_url_attr):
                link = tnode[css_url_attr]
        # normalize possible list[str] from BS4 attributes
        if isinstance(link, list):
            link = link[0] if link else None
        # join relative links
        if isinstance(link, str) and link.startswith("/"):
            # best-effort: join with base
            from urllib.parse import urljoin
            link = urljoin(url, link)
        # date
        dt = None
        if css_date:
            dnode = node.select_one(css_date)
            if dnode:
                dt_attr = dnode.get("datetime") or dnode.get_text(" ", strip=True)
                dt = norm_dt(dt_attr)
        records.append({
            "title": title,
            "url": link or url,
            "summary": "",
            "published": dt,
        })
    logger.debug(f"fetch_html_generic: selected {len(records)} nodes from {url}")
    return records

# ----------------------------- Processing -----------------------------------

def match_rules(item: dict, source_category: str, rules: List[Rule]) -> List[Tuple[Rule, int]]:
    title = (item.get("title") or "").lower()
    summary = (item.get("summary") or "").lower()
    text = f"{title} {summary}"
    matched = []
    for r in rules:
        if source_category not in r.categories:
            continue
        for p in r.patterns:
            if p.lower() in text:
                matched.append((r, r.weight))
                break
    return matched

def normalize_item(raw: dict, source_name: str, category: str) -> dict:
    title = raw.get("title") or ""
    url = raw.get("url") or ""
    published = raw.get("published")
    if isinstance(published, datetime):
        published_iso = published.isoformat()
    elif published:
        _parsed = None
        try:
            _parsed = norm_dt(published)
        except Exception:
            _parsed = None
        published_iso = _parsed.isoformat() if isinstance(_parsed, datetime) else None
    else:
        published_iso = None
    return {
        "id": sha1(f"{title}|{url}|{category}"),
        "title": title,
        "url": url,
        "source": source_name,
        "published": published_iso,
        "category": category,
        "summary": raw.get("summary") or ""
    }

def within_lookback(dt_iso: Optional[str], days: int) -> bool:
    if not dt_iso:
        return True  # допускаем без даты
    dt = norm_dt(dt_iso)
    if not dt:
        return True
    cutoff = datetime.now(dt.tzinfo or timezone.utc) - timedelta(days=days)
    return dt >= cutoff

def dedupe(records: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for r in records:
        k = r["id"]
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out

def apply_scoring(records: List[dict], rules: List[Rule]) -> Tuple[List[Event], int, int]:
    events: List[Event] = []
    risk, growth = 0, 0
    for r in records:
        matches = match_rules(r, r["category"], rules)
        if not matches:
            continue
        # Приоритет: если среди совпадений есть явное положительное "License Approval" — берём его,
        # чтобы репортажи о разрешении FAA не перекрывались общими негативами типа "Setback".
        la = None
        for rl, w in matches:
            if rl.name.lower() == "license approval" or rl.tag.lower() == "license approval":
                la = (rl, w)
                break
        if la is not None:
            rule, weight = la
        else:
            # иначе — по модулю веса
            rule, weight = sorted(matches, key=lambda t: abs(t[1]), reverse=True)[0]
        impact = rule.impact
        if impact == "risk":
            risk += weight
        else:
            growth += weight
        events.append(Event(
            id=r["id"],
            title=r["title"],
            url=r["url"],
            source=r["source"],
            published=r["published"],
            category=r["category"],
            tag=rule.tag,
            weight=weight,
            impact=impact
        ))
    # sort by abs weight desc, then date desc
    events.sort(key=lambda e: (abs(e.weight), e.published or ""), reverse=True)
    # Агрегация дублей по конкурентам Kuiper: максимум одно событие в сутки на тэг
    grouped: Dict[str, Event] = {}
    for e in events:
        day = (e.published or "")[:10]
        if e.category.startswith("Competitors/Kuiper"):
            key = f"KUIPER|{e.tag}|{day}"
        elif e.tag == "Tender Up":
            # Суточная дедупликация однотипных новостей о росте оценки/раундах
            key = f"TAG|{e.tag}|{day}"
        else:
            key = f"{e.id}"
        if key not in grouped:
            grouped[key] = e
        else:
            # сохраняем более "сильное" событие (по модулю веса), остальное дропаем
            if abs(e.weight) > abs(grouped[key].weight):
                grouped[key] = e
    events_final = list(grouped.values())

    # Общая дедупликация по дню и отпечатку заголовка (сквозь категории),
    # чтобы убирать дубли вроде NASA/General vs NASA/HLS.
    by_fp: Dict[str, Event] = {}
    def _is_hls(ev: Event) -> bool:
        t = f"{ev.title} {ev.tag} {ev.category}".lower()
        return ("hls" in t) or ("human landing" in t)
    for e in events_final:
        day = (e.published or "")[:10]
        fp = _title_fingerprint(e.title)
        key = f"{day}|{fp}"
        prev = by_fp.get(key)
        if prev is None:
            by_fp[key] = e
        else:
            # Выбираем по приоритетам: |вес|, дата, затем предпочтение HLS
            if abs(e.weight) > abs(prev.weight):
                by_fp[key] = e
            elif abs(e.weight) == abs(prev.weight):
                d_prev = prev.published or ""
                d_cur = e.published or ""
                if d_cur > d_prev:
                    by_fp[key] = e
                elif d_cur == d_prev:
                    if _is_hls(e) and not _is_hls(prev):
                        by_fp[key] = e
    events_final = list(by_fp.values())
    # Доп. дедупликация Finance/Tender по ключу оценки (сквозь дни)
    ft_by_val: Dict[str, Event] = {}
    remainder: List[Event] = []
    for e in events_final:
        if e.category.startswith("Finance/Tender"):
            vk = _valuation_key(e.title)
            if vk:
                prev = ft_by_val.get(vk)
                if not prev:
                    ft_by_val[vk] = e
                else:
                    # берём более свежую/сильную
                    d_prev = (prev.published or "")
                    d_cur = (e.published or "")
                    if d_cur > d_prev or (d_cur == d_prev and abs(e.weight) > abs(prev.weight)):
                        ft_by_val[vk] = e
                continue
        remainder.append(e)
    events_final = remainder + list(ft_by_val.values())
    # пересчёт итогов по сгруппированным событиям
    risk_final = sum(e.weight for e in events_final if e.weight < 0)
    growth_final = sum(e.weight for e in events_final if e.weight > 0)
    events_final.sort(key=lambda e: (abs(e.weight), e.published or ""), reverse=True)
    logger.debug(f"apply_scoring: events={len(events_final)} (was {len(events)}), risk={risk_final}, growth={growth_final}")
    return events_final, risk_final, growth_final

# ----------------------------- Recommendation ------------------------------

def compute_recommendation(cfg: dict, total: int) -> Tuple[str, str]:
    th = cfg.get("action_thresholds", {})
    strong_buy = th.get("strong_buy", 25)
    buy = th.get("buy", 15)
    hold = th.get("hold", 5)
    trim = th.get("trim", -10)
    reduce = th.get("reduce", -20)
    if total >= strong_buy:
        return "УВЕЛИЧИВАТЬ (сильно)", f"Индекс {total} ≥ {strong_buy}"
    elif total >= buy:
        return "УВЕЛИЧИВАТЬ", f"Индекс {total} ≥ {buy}"
    elif total >= hold:
        return "ДЕРЖАТЬ", f"Индекс {total} ≥ {hold}"
    elif total > trim:
        return "НЕЙТРАЛЬНО", f"Индекс {total} > {trim}"
    elif total > reduce:
        return "ПОДРЕЗАТЬ 10–20%", f"Индекс {total} ≤ {trim}"
    else:
        return "СНИЖАТЬ 30–50%", f"Индекс {total} ≤ {reduce}"

# ----------------------- GPT-based interpretation --------------------------

def collect_records(cfg_path: str, days: Optional[int] = None) -> Tuple[List[dict], dict]:
    """Собирает и нормализует записи без интерпретации/скоринга.
    Возвращает (records, cfg).
    """
    logger.debug(f"collect_records: load cfg from {cfg_path}")
    cfg = load_yaml(cfg_path)
    lookback_days = int(days or cfg.get("lookback_days", 30))
    db_path = cfg.get("database_path", "data/articles.jsonl")
    records_all: List[dict] = []

    sources = cfg.get("sources", [])
    logger.debug(f"collect_records: sources={len(sources)}")
    for s in sources:
        if not s.get("url"):
            continue
        if s.get("enabled") is False:
            logger.debug(f"source disabled: {s.get('name')}")
            continue
        name = s.get("name")
        stype = s.get("type")
        url = s.get("url")
        category = s.get("category", "Other")
        try:
            logger.debug(f"fetch: {name} [{stype}] {url}")
            if stype == "rss":
                items = fetch_rss(url)
            elif stype == "html":
                items = fetch_html_generic(
                    url,
                    s.get("css_item") or "article",
                    s.get("css_title") or "h1,h2,h3",
                    s.get("css_url_selector"),
                    s.get("css_url_attr") or "href",
                    s.get("css_date"),
                )
            else:
                items = []
        except Exception as e:
            console.print(f"[yellow]Источник {name} не собран: {e}[/yellow]")
            items = []
        logger.debug(f"fetched {len(items)} items from {name}")
        for it in items:
            rec = normalize_item(it, name, category)
            if within_lookback(rec["published"], lookback_days):
                records_all.append(rec)

    logger.debug(f"collect_records: records_all={len(records_all)} before dedupe")
    records = dedupe(records_all)
    logger.debug(f"collect_records: records={len(records)} after dedupe")

    # save raw to DB (append)
    save_jsonl(db_path, records)
    logger.debug(f"collect_records: saved {len(records)} records to {db_path}")
    return records, cfg

def gpt_score_records(records: List[dict], model: str, max_items: int = 40, timeout: float = 20.0, cache_path: str = "data/gpt_cache.jsonl") -> Tuple[List[Event], int, int]:
    """
    Интерпретируем новости с помощью GPT. Для каждой записи возвращаем impact (risk/growth/neutral), вес [-10..10], tag.
    Используем кэш по id, чтобы не дергать API повторно.
    """
    if OpenAI is None:
        raise RuntimeError("openai пакет недоступен. Установите openai>=1.40.0")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не задан в окружении")
    # Подсказка: fine-grained/restricted ключи (rk-...) требуют скоуп model.request
    if api_key.startswith("rk-"):
        logger.warning("Обнаружен restricted ключ OpenAI (rk-...). Убедитесь, что ему назначен scope 'model.request' и доступ к модели.")
    if api_key.startswith("sk-proj-") and not os.getenv("OPENAI_PROJECT"):
        logger.warning("Обнаружен project-ключ OpenAI (sk-proj-...), но переменная OPENAI_PROJECT не задана. Рекомендуется указать идентификатор проекта OPENAI_PROJECT=proj_... для корректной авторизации.")

    base_url = os.getenv("OPENAI_BASE_URL") or None
    project = os.getenv("OPENAI_PROJECT") or None

    ensure_dir(os.path.dirname(cache_path))
    cache_map = load_jsonl_map(cache_path)

    # фильтруем только новые для GPT
    pending: List[dict] = []
    for r in records[:max_items]:
        if r["id"] not in cache_map:
            pending.append(r)

    logger.info(f"GPT: всего записей {len(records)}, к интерпретации {len(pending)}, кэш {len(cache_map)}")

    # Инициализация клиента с учётом project/base_url при наличии
    try:
        if base_url and project:
            client = OpenAI(timeout=timeout, base_url=base_url, project=project, api_key=api_key)  # type: ignore
        elif project:
            client = OpenAI(timeout=timeout, project=project, api_key=api_key)  # type: ignore
        elif base_url:
            client = OpenAI(timeout=timeout, base_url=base_url, api_key=api_key)  # type: ignore
        else:
            client = OpenAI(timeout=timeout, api_key=api_key)  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Не удалось инициализировать OpenAI клиент: {e}")

    def chunks(lst: List[dict], n: int) -> List[List[dict]]:
        return [lst[i:i+n] for i in range(0, len(lst), n)]

    # Бьем на чанки по 20 для стабильности
    for batch in chunks(pending, 20):
        payload = [
            {
                "id": r["id"],
                "title": r.get("title"),
                "summary": r.get("summary"),
                "category": r.get("category"),
                "source": r.get("source"),
                "published": r.get("published"),
            }
            for r in batch
        ]
        sys_msg = (
            "Ты финансовый аналитик космической отрасли. Оцени новости о SpaceX по влиянию на акцию компании. "
            "Всегда отвечай на русском языке. Для каждого элемента верни: impact ('growth'|'risk'|'neutral'), "
            "целочисленный weight от -10 до 10, краткий tag на русском (1-3 слова), и короткий заголовок на русском title_ru (до 90 символов). "
            "Учитывай категорию, источник, заголовок и дату. Возвращай только JSON-объект."
        )
        user_msg = {
            "instruction": "Оцени элементы и верни JSON вида {\"items\":[{\"id\":...,\"impact\":...,\"weight\":...,\"tag\":...,\"title_ru\":...}]}.",
            "items": payload,
        }
        items: List[dict] = []
        for attempt in range(2):
            try:
                logger.debug(f"GPT: отправка батча, size={len(batch)}, attempt={attempt+1}")
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                content = resp.choices[0].message.content  # type: ignore
                data = json.loads(content or "{}")
                items = cast(List[dict], data.get("items", []))
                logger.debug(f"GPT: получено {len(items)} оценок")
                break
            except Exception as e:
                msg = str(e)
                if "Missing scopes" in msg or "insufficient permissions" in msg:
                    logger.warning("GPT: ошибка запроса (нет прав/скоупов): %s", msg)
                    break
                logger.warning(f"GPT: ошибка запроса: {e}")
                if attempt == 0:
                    time.sleep(2)
                else:
                    items = []

        # Сохраняем в кэш
        to_cache = []
        for it in items:
            k = it.get("id")
            if isinstance(k, str):
                cache_map[k] = it
                to_cache.append({"id": k, **{kk: vv for kk, vv in it.items() if kk != "id"}})
        if to_cache:
            save_jsonl(cache_path, to_cache)
            logger.debug(f"GPT: кэшировано {len(to_cache)} записей в {cache_path}")

    # Заранее построим карту отпечатков оригинальных заголовков для дедупликации
    orig_fp_map: Dict[str, str] = {}
    for r in records[:max_items]:
        try:
            orig_fp_map[r["id"]] = _title_fingerprint(r.get("title") or "")
        except Exception:
            pass

    # Формируем события из кэша (используем только записи с ненулевым весом)
    events: List[Event] = []
    risk, growth = 0, 0
    for r in records[:max_items]:
        it = cache_map.get(r["id"])
        if not isinstance(it, dict):
            continue
        impact = str(it.get("impact") or "neutral").lower()
        try:
            weight = int(it.get("weight", 0))
        except Exception:
            weight = 0
        tag = str(it.get("tag") or "—")
        title_ru = str(it.get("title_ru") or r["title"])[:200]
        if weight == 0 or impact == "neutral":
            continue
        if impact == "risk":
            risk += abs(weight) if weight < 0 else weight  # в нашем индексе риск хранится как положительный вклад риска
            # но итоговый total = risk + growth, где risk может быть отриц. В исходнике risk суммируется весами правила.
            # Для совместимости: считаем risk как отрицательный вклад, growth как положительный, как было ранее.
            # Поэтому приведем знак:
            w = -abs(weight)
        else:
            w = abs(weight)
            growth += w
        events.append(Event(
            id=r["id"],
            title=title_ru,
            url=r["url"],
            source=r["source"],
            published=r["published"],
            category=r["category"],
            tag=tag,
            weight=w,
            impact="risk" if w < 0 else "growth",
        ))

    # Дедупликация для GPT: группируем по дню + отпечатку исходного заголовка,
    # берём событие с максимальным |весом|, при равенстве — более свежее, затем предпочитаем HLS.
    grouped: Dict[str, Event] = {}
    def _is_hls(ev: Event) -> bool:
        t = f"{ev.title} {ev.tag} {ev.category}".lower()
        return ("hls" in t) or ("human landing" in t)
    for e in events:
        day = (e.published or "")[:10]
        fp = orig_fp_map.get(e.id) or _title_fingerprint(e.title)
        key = f"{day}|{fp}"
        prev = grouped.get(key)
        if prev is None:
            grouped[key] = e
        else:
            if abs(e.weight) > abs(prev.weight):
                grouped[key] = e
            elif abs(e.weight) == abs(prev.weight):
                d_prev = prev.published or ""
                d_cur = e.published or ""
                if d_cur > d_prev:
                    grouped[key] = e
                elif d_cur == d_prev:
                    if _is_hls(e) and not _is_hls(prev):
                        grouped[key] = e
    deduped = list(grouped.values())

    # Доп. дедупликация для Finance/Tender: предпочитаем ключ оценки, затем отпечаток заголовка
    ft_by_val: Dict[str, Event] = {}
    ft_by_fp: Dict[str, Event] = {}
    rest: List[Event] = []
    for e in deduped:
        if e.category.startswith("Finance/Tender"):
            vk = _valuation_key(e.title)
            if vk:
                prev = ft_by_val.get(vk)
                if not prev:
                    ft_by_val[vk] = e
                else:
                    d_prev = (prev.published or "")
                    d_cur = (e.published or "")
                    if d_cur > d_prev or (d_cur == d_prev and abs(e.weight) > abs(prev.weight)):
                        ft_by_val[vk] = e
                continue
            fp = _title_fingerprint(e.title)
            prev = ft_by_fp.get(fp)
            if not prev:
                ft_by_fp[fp] = e
            else:
                d_prev = (prev.published or "")
                d_cur = (e.published or "")
                if d_cur > d_prev or (d_cur == d_prev and abs(e.weight) > abs(prev.weight)):
                    ft_by_fp[fp] = e
            continue
        rest.append(e)
    deduped = rest + list(ft_by_val.values()) + list(ft_by_fp.values())

    # Сортировка схожа с rule-based
    deduped.sort(key=lambda e: (abs(e.weight), e.published or ""), reverse=True)
    logger.debug(f"GPT: events={len(events)} -> deduped={len(deduped)}, risk={sum(e.weight for e in deduped if e.weight<0)}, growth={sum(e.weight for e in deduped if e.weight>0)}")

    # Пересчитаем суммарные risk/growth
    risk_sum = sum(e.weight for e in deduped if e.weight < 0)
    growth_sum = sum(e.weight for e in deduped if e.weight > 0)
    return deduped, risk_sum, growth_sum

# ------------------------- Summary Builder (shared) -------------------------

def build_summary(cfg_path: str, checklist_path: str, days: Optional[int], force_gpt: bool = True,
                  gpt_model: str = "gpt-4o-mini", gpt_timeout: float = 20.0, gpt_max_items: int = 40,
                  gpt_cache: str = "data/gpt_cache.jsonl") -> Tuple[List[Event], int, int, List[Tuple[str,int]], dict]:
    """Единое построение сводки: собирает записи и возвращает (events, risk, growth, checklist, cfg).
    По умолчанию использует GPT при наличии ключа или если force_gpt=True.
    """
    cfg = load_yaml(cfg_path)
    use_gpt = force_gpt and bool(os.getenv("OPENAI_API_KEY"))
    checklist_applied: List[Tuple[str,int]] = []
    if use_gpt:
        records, cfg = collect_records(cfg_path, days)
        events, risk, growth = gpt_score_records(
            records=records,
            model=gpt_model,
            max_items=gpt_max_items,
            timeout=gpt_timeout,
            cache_path=gpt_cache,
        )
    else:
        events, risk, growth, checklist_applied = collect(cfg_path, checklist_path, days)
    # применим чеклист (как в main)
    if os.path.exists(checklist_path):
        ck = load_yaml(checklist_path)
        mdefs = cfg.get("manual_checklist", {})
        for key, val in ck.items():
            if val and key in mdefs:
                w = int(mdefs[key].get("weight", 0))
                if w != 0:
                    checklist_applied.append((key, w))
                    if w > 0:
                        growth += w
                    else:
                        risk += w
    # применим затухание, чтобы все потребители (HTML/Telegram/бот) видели одинаковый набор и веса
    decay_days = int(cfg.get("decay_days", 7))
    if decay_days and decay_days > 0:
        events = apply_decay(events, decay_days)
        risk = sum(e.weight for e in events if e.weight < 0)
        growth = sum(e.weight for e in events if e.weight > 0)
    return events, risk, growth, checklist_applied, cfg

# --------------------------- USAspending API --------------------------------

def usaspending_pull(recipient_name: str, lookback_days: int) -> List[dict]:
    """
    Тянем новые контракты/обязательства за lookback_days.
    """
    base = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
    end = datetime.utcnow().date()
    start = end - timedelta(days=lookback_days)
    payload = {
        "fields": [
            "Award ID", "Description", "Recipient Name", "Award Amount",
            "Action Date", "Awarding Agency"
        ],
        "filters": {
            "time_period": [{"start_date": start.isoformat(), "end_date": end.isoformat()}],
            "recipient_search_text": [recipient_name],
            "award_type_codes": ["A", "B", "C", "D"],
        },
        "page": 1,
        "limit": 100,
        "sort": "Action Date",
        "order": "desc",
    }
    try:
        resp = requests.post(base, json=payload, headers=HEADERS, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
    except Exception as e:
        logger.debug(f"USAspending запрос не удался: {e}")
        results = []
    out = []
    for r in results:
        # Устойчивое извлечение полей с разными схемами ключей
        title = (
            r.get("Description")
            or r.get("description")
            or r.get("piid")
            or r.get("Award ID")
            or r.get("award_id")
            or "Award"
        )
        url = None  # TODO: можно построить ссылку usaspending.gov/award/<id> если пришёл unique_award_key
        published = r.get("Action Date") or r.get("action_date")
        agency = r.get("Awarding Agency") or r.get("awarding_agency") or r.get("awarding_agency_name") or ""
        amount = r.get("Award Amount") or r.get("award_amount") or r.get("obligated_amount") or 0
        out.append({
            "title": f"USAspending: {title} ({agency}, ${amount:,.0f})",
            "url": url or "https://www.usaspending.gov/",
            "summary": "",
            "published": norm_dt(published),
            "source": "USAspending API",
            "category": "DoD/Contracts" if "Defense" in agency or "Space Force" in agency else "NASA/HLS"
        })
    return out

# ------------------------------ Rendering -----------------------------------

def render_dashboard(cfg: dict, events: List[Event], risk: int, growth: int, checklist_applied: List[Tuple[str, int]], out_path: str) -> None:
    env = Environment(
        loader=FileSystemLoader(searchpath="templates"),
        autoescape=select_autoescape(['html', 'xml'])
    )
    tpl = env.get_template("dashboard.html.j2")
    total = risk + growth
    # checklist summary
    if checklist_applied:
        ck_summary = " + ".join([f"{name}({('%+d'%w)})" for name, w in checklist_applied])
    else:
        ck_summary = "—"
    # recommendation
    action, reason = compute_recommendation(cfg, total)

    html = tpl.render(
        generated_at=utcnow_iso(),
        lookback_days=cfg.get("lookback_days", 30),
        source_count=len(cfg.get("sources", [])),
        total_score=total,
        risk_score=risk,
        growth_score=growth,
        events=[e.__dict__ for e in events],
        recommendation={"action": action, "reason": reason},
        checklist_summary=ck_summary,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    console.print(f"[green]Сохранено:[/green] {out_path}")

# ---------------------------- Telegram Notify ------------------------------

def send_telegram_message(token: str, chat_id: str, text: str, parse_mode: str = "HTML", reply_markup: Optional[dict] = None) -> Tuple[bool, Optional[str]]:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": False}
        if reply_markup:
            payload["reply_markup"] = reply_markup
        r = requests.post(url, json=payload, timeout=20)
        ok = False
        err = None
        try:
            data = r.json()
            ok = bool(data.get("ok", False))
            if not ok:
                err = json.dumps(data, ensure_ascii=False)
        except Exception:
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        return ok, err
    except Exception as e:
        return False, str(e)

def build_telegram_text(cfg: dict, events: List[Event], risk: int, growth: int) -> str:
    total = risk + growth
    action, reason = compute_recommendation(cfg, total)
    parts = []
    parts.append(f"<b>SpaceX Monitor</b> — индекс: <b>{total:+d}</b> (рост {growth:+d}, риск {risk:+d})")
    parts.append(f"Рекомендация: <b>{action}</b> — {reason}")
    if events:
        # Разделим на позитивные и негативные
        pos = sorted([e for e in events if e.weight > 0], key=lambda x: x.weight, reverse=True)
        neg = sorted([e for e in events if e.weight < 0], key=lambda x: abs(x.weight), reverse=True)
        if pos:
            parts.append("\n<b>Топ позитивные:</b>")
            for e in pos[:5]:
                w = e.weight
                title = e.title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                d = (e.published or "")[:10]
                parts.append(f"{d} · +{w} · <a href=\"{e.url}\">{title}</a> [{e.tag}]")
        if neg:
            parts.append("\n<b>Топ негативные:</b>")
            for e in neg[:5]:
                w = abs(e.weight)
                title = e.title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                d = (e.published or "")[:10]
                parts.append(f"{d} · −{w} · <a href=\"{e.url}\">{title}</a> [{e.tag}]")
    parts.append(f"\nОбновлено: {utcnow_iso()}")
    return "\n".join(parts)

def save_events_csv(events: List[Event], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "title", "url", "source", "category", "tag", "weight", "impact"])
        for e in events:
            writer.writerow([e.published, e.title, e.url, e.source, e.category, e.tag, e.weight, e.impact])
    console.print(f"[green]CSV сохранён:[/green] {out_path}")

# ------------------------------ Decay logic ---------------------------------

def apply_decay(events: List[Event], decay_days: int) -> List[Event]:
    """Применяет линейное затухание веса события по возрасту публикации.
    - decay_days > 0: factor = max(0, 1 - age_days / decay_days)
    - decay_days <= 0: без затухания
    Возвращает новый список, исключая события с нулевым итоговым весом.
    """
    if decay_days is None or decay_days <= 0:
        return events
    now = datetime.now(timezone.utc)
    out: List[Event] = []
    for e in events:
        dt = norm_dt(e.published) if e.published else None
        if not isinstance(dt, datetime):
            # без даты — без затухания
            out.append(e)
            continue
        # нормализуем таймзону
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
        factor = max(0.0, 1.0 - (age_days / float(decay_days)))
        new_w = int(round(abs(e.weight) * factor))
        if new_w <= 0:
            continue
        new_w = -new_w if e.weight < 0 else new_w
        out.append(Event(
            id=e.id,
            title=e.title,
            url=e.url,
            source=e.source,
            published=e.published,
            category=e.category,
            tag=e.tag,
            weight=new_w,
            impact=("risk" if new_w < 0 else "growth"),
        ))
    # пересортируем по |весу| и дате
    out.sort(key=lambda x: (abs(x.weight), x.published or ""), reverse=True)
    return out

# ------------------------------ Main logic ----------------------------------

def collect(cfg_path: str, checklist_path: str, days: Optional[int] = None) -> Tuple[List[Event], int, int, List[Tuple[str,int]]]:
    logger.debug(f"collect: load cfg from {cfg_path}")
    cfg = load_yaml(cfg_path)
    lookback_days = int(days or cfg.get("lookback_days", 30))
    logger.debug(f"collect: lookback_days={lookback_days}")
    db_path = cfg.get("database_path", "data/articles.jsonl")
    records_all: List[dict] = []

    # load rules
    rules_cfg = cfg.get("rules", [])
    rules: List[Rule] = [Rule(
        name=r["name"],
        patterns=r["patterns"],
        categories=r["categories"],
        weight=int(r["weight"]),
        tag=r["tag"],
        impact=r["impact"]
    ) for r in rules_cfg]
    logger.debug(f"collect: loaded rules={len(rules)}")

    # fetch each source
    sources = cfg.get("sources", [])
    logger.debug(f"collect: sources={len(sources)}")
    for s in sources:
        if not s.get("url"):
            continue
        if s.get("enabled") is False:
            logger.debug(f"source disabled: {s.get('name')}")
            continue
        name = s.get("name")
        stype = s.get("type")
        url = s.get("url")
        category = s.get("category", "Other")
        try:
            logger.debug(f"fetch: {name} [{stype}] {url}")
            if stype == "rss":
                items = fetch_rss(url)
            elif stype == "html":
                items = fetch_html_generic(
                    url,
                    s.get("css_item") or "article",
                    s.get("css_title") or "h1,h2,h3",
                    s.get("css_url_selector"),
                    s.get("css_url_attr") or "href",
                    s.get("css_date"),
                )
            else:
                items = []
        except Exception as e:
            console.print(f"[yellow]Источник {name} не собран: {e}[/yellow]")
            items = []
        # normalize
        logger.debug(f"fetched {len(items)} items from {name}")
        for it in items:
            rec = normalize_item(it, name, category)
            if within_lookback(rec["published"], lookback_days):
                records_all.append(rec)
    logger.debug(f"collect: records_all={len(records_all)} before dedupe")

    # extra: USAspending new awards
    if cfg.get("usaspending_enabled", True):
        try:
            usa_items = usaspending_pull("SPACE EXPLORATION TECHNOLOGIES CORP.", cfg.get("usaspending_lookback_days", 120))
            for it in usa_items:
                rec = {
                    "id": sha1(f"{it['title']}|{it['url']}|{it['category']}"),
                    "title": it["title"],
                    "url": it["url"],
                    "source": "USAspending API",
                    "published": it["published"].isoformat() if isinstance(it["published"], datetime) else (it["published"].isoformat() if it["published"] else None),
                    "category": it["category"],
                    "summary": it.get("summary") or ""
                }
                if within_lookback(rec["published"], lookback_days):
                    records_all.append(rec)
        except Exception as e:
            console.print(f"[yellow]USAspending ошибка: {e}[/yellow]")

    # dedupe
    records = dedupe(records_all)
    logger.debug(f"collect: records={len(records)} after dedupe")

    # save raw to DB
    save_jsonl(db_path, records)
    logger.debug(f"collect: saved {len(records)} records to {db_path}")

    # score (rule-based)
    events, risk, growth = apply_scoring(records, rules)

    # manual checklist
    checklist_applied: List[Tuple[str, int]] = []
    if os.path.exists(checklist_path):
        ck = load_yaml(checklist_path)
        mdefs = cfg.get("manual_checklist", {})
        for key, val in ck.items():
            if val and key in mdefs:
                w = int(mdefs[key].get("weight", 0))
                if w != 0:
                    checklist_applied.append((key, w))
                    if w > 0:
                        growth += w
                    else:
                        risk += w

    logger.debug(f"collect: final risk={risk}, growth={growth}, total={risk+growth}, events={len(events)}")
    return events, risk, growth, checklist_applied

def main():
    parser = argparse.ArgumentParser(description="SpaceX Monitoring Dashboard")
    parser.add_argument("command", choices=["run", "report"], help="run — собрать и построить дашборд; report — краткий отчёт в консоль")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--checklist", default="checklist.yaml")
    parser.add_argument("--days", type=int, default=None, help="окно (дни), переопределяет lookback_days")
    parser.add_argument("--debug", action="store_true", help="включить подробное логирование (DEBUG)")
    # GPT режим (включен по умолчанию)
    parser.add_argument("--use-gpt", action="store_true", help="использовать GPT для интерпретации вместо правил")
    parser.add_argument("--gpt-model", default="gpt-4o-mini", help="модель OpenAI для интерпретации")
    parser.add_argument("--gpt-timeout", type=float, default=20.0, help="таймаут запроса к OpenAI, сек")
    parser.add_argument("--gpt-max-items", type=int, default=40, help="сколько последних записей интерпретировать (для скорости)")
    parser.add_argument("--gpt-cache", default="data/gpt_cache.jsonl", help="путь к кэшу интерпретаций GPT")
    parser.add_argument("--gpt-strict", action="store_true", help="только GPT: при ошибке или пустом ответе завершить без фоллбэка на правила")
    # Telegram
    parser.add_argument("--telegram", action="store_true", help="отправить уведомление в Telegram")
    parser.add_argument("--no-telegram", action="store_true", help="не отправлять уведомление, даже если заданы переменные окружения")
    parser.add_argument("--tg-token", default=None, help="Telegram Bot token (иначе TELEGRAM_BOT_TOKEN из окружения)")
    parser.add_argument("--tg-chat-id", default=None, help="Telegram chat_id (иначе TELEGRAM_CHAT_ID из окружения)")
    # Serving dashboard as a mini web app
    parser.add_argument("--serve", action="store_true", help="поднять простой HTTP-сервер для каталога вывода (dashboard.html)")
    parser.add_argument("--host", default="127.0.0.1", help="хост для HTTP-сервера (по умолчанию 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="порт для HTTP-сервера (по умолчанию 8000)")
    parser.add_argument("--public-url", default=None, help="публичный базовый URL до каталога вывода (напр., https://your-domain.com/monitor_output/). Если задан, в Telegram добавится кнопка открытия дашборда как мини‑приложения.")
    args = parser.parse_args()

    # Поддержка строгого режима через переменную окружения
    if (os.getenv("GPT_STRICT", "").strip().lower() in ("1", "true", "yes", "on")):
        args.gpt_strict = True

    # logging setup
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Урезаем болтливость внешних HTTP-библиотек, чтобы не светить чувствительные URL (например, Telegram токен)
    for noisy in ["urllib3", "httpx", "openai", "httpcore", "requests"]:
        try:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        except Exception:
            pass
    logger.debug("debug logging enabled")

    # Основной поток: строим сводку единым способом
    try:
        events, risk, growth, checklist_applied, cfg = build_summary(
            cfg_path=args.config,
            checklist_path=args.checklist,
            days=args.days,
            force_gpt=(True if os.getenv("OPENAI_API_KEY") else args.use_gpt),
            gpt_model=args.gpt_model,
            gpt_timeout=args.gpt_timeout,
            gpt_max_items=args.gpt_max_items,
            gpt_cache=args.gpt_cache,
        )
        # Если строгий GPT, но событий нет — завершаем
        if args.gpt_strict and not events and (risk == 0 and growth == 0):
            console.print("[red]GPT не вернул оценок. Режим --gpt-strict: завершаем без фоллбэка.[/red]")
            sys.exit(3)
    except Exception as e:
        if args.gpt_strict:
            console.print(f"[red]Ошибка при построении сводки в GPT-режиме: {e}[/red]")
            sys.exit(2)
        console.print(f"[yellow]Не удалось построить GPT-сводку ({e}). Используем правила.[/yellow]")
        events, risk, growth, checklist_applied = collect(args.config, args.checklist, args.days)
        cfg = load_yaml(args.config)
    total = risk + growth

    # Применим затухание согласно конфигу (по умолчанию 7 дней, если указано)
    decay_days = int(load_yaml(args.config).get("decay_days", 7))
    if decay_days and decay_days > 0:
        events = apply_decay(events, decay_days)
        # Пересчёт risk/growth
        risk = sum(e.weight for e in events if e.weight < 0)
        growth = sum(e.weight for e in events if e.weight > 0)
    # Телеграм: авто-отправка, если заданы переменные окружения и не отключено, или по флагу --telegram
    want_tg = (args.telegram or (not args.no_telegram and os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")))
    if want_tg:
        cfg_for_rec = load_yaml(args.config)
        msg = build_telegram_text(cfg_for_rec, events, risk, growth)
        tg_token = args.tg_token or os.getenv("TELEGRAM_BOT_TOKEN")
        tg_chat = args.tg_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        # Формируем URL дашборда, если доступен
        dash_url: Optional[str] = None
        if args.public_url:
            base = args.public_url if args.public_url.endswith("/") else args.public_url + "/"
            dash_url = urljoin(base, "dashboard.html")
        elif args.serve:
            scheme = "http"
            dash_url = f"{scheme}://{args.host}:{args.port}/dashboard.html"
        # Соберём inline-клавиатуру. Если TELEGRAM_USE_WEBAPP=1 и есть HTTPS-URL, используем web_app, иначе обычная url-кнопка.
        reply_markup = None
        if dash_url:
            use_webapp = os.getenv("TELEGRAM_USE_WEBAPP", "").strip().lower() in ("1", "true", "yes", "on") and dash_url.startswith("https://")
            button: Dict[str, Any]
            if use_webapp:
                button = {"text": "Открыть дашборд", "web_app": {"url": dash_url}}
            else:
                button = {"text": "Открыть дашборд", "url": dash_url}
            reply_markup = {"inline_keyboard": [[button]]}
        if tg_token and tg_chat:
            ok, err = send_telegram_message(tg_token, tg_chat, msg, reply_markup=reply_markup)
            if ok:
                console.print("[green]Telegram уведомление отправлено.[/green]")
            else:
                console.print(f"[yellow]Не удалось отправить в Telegram: {err}[/yellow]")
        else:
            console.print("[yellow]Telegram: отсутствуют токен или chat_id. Пропускаем отправку.[/yellow]")

    if args.command == "report":
        table = Table(title=f"SpaceX Monitor: индекс={total} (риск {risk}, рост {growth})", box=box.SIMPLE_HEAVY)
        table.add_column("Дата", style="cyan", no_wrap=True)
        table.add_column("Заголовок", style="white")
        table.add_column("Категория", style="magenta")
        table.add_column("Тэг", style="green")
        table.add_column("Вес", style="bold")
        for e in events[:50]:
            table.add_row((e.published or "")[:10], e.title, e.category, e.tag, f"{e.weight:+d}")
        console.print(table)
        if checklist_applied:
            console.print(f"[blue]Чеклист:[/blue] " + ", ".join(f"{k}({('%+d'%w)})" for k, w in checklist_applied))
        return

    # command == run
    out_dir = load_yaml(args.config).get("output_dir", "monitor_output")
    ensure_dir(out_dir)
    render_dashboard(load_yaml(args.config), events, risk, growth, checklist_applied, os.path.join(out_dir, "dashboard.html"))
    save_events_csv(events, os.path.join(out_dir, "events.csv"))
    # also save a machine-friendly summary
    summary = {
        "generated_at": utcnow_iso(),
        "risk": risk,
        "growth": growth,
        "total": total,
        "events": [e.__dict__ for e in events],
        "checklist_applied": checklist_applied,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    console.print(f"[bold green]Готово.[/bold green] Открой файл {os.path.join(out_dir, 'dashboard.html')} в браузере.")

    # Если попросили, поднимем простой HTTP-сервер для каталога вывода
    if args.serve and args.command == "run":
        import http.server
        import socketserver
        from functools import partial

        # Поддержка автопорта Heroku ($PORT)
        port_env = os.getenv("PORT")
        port = int(port_env) if port_env and port_env.isdigit() else args.port
        host = args.host

        directory = os.path.abspath(out_dir)

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *hargs, **hkw):
                super().__init__(*hargs, directory=directory, **hkw)

            def do_GET(self):
                if self.path in ("/", ""):
                    self.send_response(302)
                    self.send_header("Location", "/dashboard.html")
                    self.end_headers()
                    return
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(b"ok")
                    return
                return super().do_GET()

        with socketserver.TCPServer((host, port), Handler) as httpd:
            url_local = f"http://{host}:{port}/dashboard.html"
            console.print(f"[blue]HTTP-сервер запущен:[/blue] {url_local}")
            if args.public_url:
                base = args.public_url if args.public_url.endswith("/") else args.public_url + "/"
                console.print(f"[blue]Публичный URL:[/blue] {urljoin(base, 'dashboard.html')}")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                console.print("Остановка HTTP-сервера…")
            finally:
                httpd.server_close()

if __name__ == "__main__":
    main()
