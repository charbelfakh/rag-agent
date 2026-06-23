"""Server-side conversation session storage (Sprint P rank 58)."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_store = None


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes")


def is_session_storage_enabled() -> bool:
    return _env_bool("SESSION_STORAGE_ENABLED")


class SessionStore:
    """Persist chat sessions and messages in SQLite for server-side history."""

    def __init__(self, db_path: str | None = None):
        self.db_path = Path(
            db_path or os.getenv("SESSION_DB_PATH", "data/sessions.db")
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at REAL,
                    updated_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    meta_json TEXT,
                    created_at REAL
                )
                """
            )
            conn.commit()
            conn.close()

    def create_session(self, title: str = "New chat") -> dict:
        session_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
            conn.commit()
            conn.close()
        return {"session_id": session_id, "title": title, "created_at": now}

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT session_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
        return [dict(row) for row in rows]

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        meta: dict | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO session_messages (session_id, role, content, meta_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, json.dumps(meta or {}), now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            conn.commit()
            conn.close()

    def get_messages(self, session_id: str) -> list[dict]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT role, content, meta_json, created_at FROM session_messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            conn.close()
        messages = []
        for row in rows:
            messages.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "meta": json.loads(row["meta_json"] or "{}"),
                    "created_at": row["created_at"],
                }
            )
        return messages


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store


def reset_session_store() -> None:
    global _store
    _store = None
