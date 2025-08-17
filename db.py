#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простая обвязка БД: subscribers и scores.
Локально — SQLite (data/monitor.db). На Heroku — Postgres (DATABASE_URL).
"""
import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional, Tuple

try:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore
except Exception:
    psycopg2 = None  # type: ignore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SQLiteDB:
    def __init__(self, path: str = "data/monitor.db") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._init()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS subscribers(
                    chat_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_seen TEXT
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS scores(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    total INTEGER,
                    risk INTEGER,
                    growth INTEGER,
                    message TEXT
                )
                """
            )

    # subscribers
    def add_subscriber(self, chat_id: int, username: Optional[str]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO subscribers(chat_id, username, first_seen) VALUES(?,?,?)",
                (chat_id, username, _now_iso()),
            )

    def get_subscribers(self) -> List[int]:
        with self._conn() as c:
            rows = c.execute("SELECT chat_id FROM subscribers").fetchall()
            return [int(r[0]) for r in rows]

    # scores
    def save_score(self, total: int, risk: int, growth: int, message: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO scores(created_at,total,risk,growth,message) VALUES(?,?,?,?,?)",
                (_now_iso(), total, risk, growth, message),
            )

    def get_latest_score(self) -> Optional[Tuple[int, int, int]]:
        with self._conn() as c:
            row = c.execute("SELECT total,risk,growth FROM scores ORDER BY id DESC LIMIT 1").fetchone()
            if not row:
                return None
            return int(row[0]), int(row[1]), int(row[2])

    def get_prev_score(self) -> Optional[Tuple[int, int, int]]:
        with self._conn() as c:
            row = c.execute("SELECT total,risk,growth FROM scores ORDER BY id DESC LIMIT 1 OFFSET 1").fetchone()
            if not row:
                return None
            return int(row[0]), int(row[1]), int(row[2])

    def get_latest_message(self) -> Optional[str]:
        with self._conn() as c:
            row = c.execute("SELECT message FROM scores ORDER BY id DESC LIMIT 1").fetchone()
            return row[0] if row else None


class PostgresDB:
    def __init__(self, dsn: str) -> None:
        if psycopg2 is None:
            raise RuntimeError("psycopg2 недоступен — добавьте psycopg2-binary в requirements.txt")
        self.dsn = dsn
        self._init()

    def _conn(self):
        # type: ignore[union-attr]
        return psycopg2.connect(self.dsn)  # type: ignore

    def _init(self) -> None:
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS subscribers(
                        chat_id BIGINT PRIMARY KEY,
                        username TEXT,
                        first_seen TIMESTAMPTZ
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scores(
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ,
                        total INTEGER,
                        risk INTEGER,
                        growth INTEGER,
                        message TEXT
                    )
                    """
                )

    # subscribers
    def add_subscriber(self, chat_id: int, username: Optional[str]) -> None:
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    "INSERT INTO subscribers(chat_id,username,first_seen) VALUES(%s,%s,now()) ON CONFLICT (chat_id) DO NOTHING",
                    (chat_id, username),
                )

    def get_subscribers(self) -> List[int]:
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute("SELECT chat_id FROM subscribers")
                return [int(r[0]) for r in cur.fetchall()]

    # scores
    def save_score(self, total: int, risk: int, growth: int, message: str) -> None:
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute(
                    "INSERT INTO scores(created_at,total,risk,growth,message) VALUES(now(),%s,%s,%s,%s)",
                    (total, risk, growth, message),
                )

    def get_latest_score(self) -> Optional[Tuple[int, int, int]]:
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute("SELECT total,risk,growth FROM scores ORDER BY id DESC LIMIT 1")
                row = cur.fetchone()
                if not row:
                    return None
                return int(row[0]), int(row[1]), int(row[2])

    def get_prev_score(self) -> Optional[Tuple[int, int, int]]:
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute("SELECT total,risk,growth FROM scores ORDER BY id DESC LIMIT 1 OFFSET 1")
                row = cur.fetchone()
                if not row:
                    return None
                return int(row[0]), int(row[1]), int(row[2])

    def get_latest_message(self) -> Optional[str]:
        with self._conn() as c:
            with c.cursor() as cur:
                cur.execute("SELECT message FROM scores ORDER BY id DESC LIMIT 1")
                row = cur.fetchone()
                return row[0] if row else None


def get_db():
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return PostgresDB(dsn)
    return SQLiteDB()
