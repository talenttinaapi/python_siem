"""Persistent SQLite-backed queue for failed webhook deliveries."""
import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import time


DEFAULT_DB = "logs/failed_webhooks.db"


def init_db(path: str | Path = DEFAULT_DB) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS queue (
            id INTEGER PRIMARY KEY,
            event TEXT NOT NULL,
            url TEXT NOT NULL,
            attempts INTEGER DEFAULT 0,
            max_attempts INTEGER DEFAULT 3,
            base_backoff REAL DEFAULT 0.5,
            next_attempt REAL DEFAULT 0,
            created REAL DEFAULT (strftime('%s','now'))
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dead_letter (
            id INTEGER PRIMARY KEY,
            event TEXT NOT NULL,
            url TEXT NOT NULL,
            attempts INTEGER,
            failed_at REAL DEFAULT (strftime('%s','now'))
        )
        """
    )
    conn.commit()
    conn.close()


def enqueue(path: str | Path, event: Dict[str, Any], url: str, max_attempts: int = 3, base_backoff: float = 0.5) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO queue (event, url, attempts, max_attempts, base_backoff, next_attempt) VALUES (?, ?, 0, ?, ?, ?)",
        (json.dumps(event), url, max_attempts, base_backoff, 0),
    )
    id = cur.lastrowid
    conn.commit()
    conn.close()
    return id


def list_due(path: str | Path, limit: int = 100) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    now = time.time()
    cur.execute("SELECT id, event, url, attempts, max_attempts, base_backoff FROM queue WHERE next_attempt<=? ORDER BY id LIMIT ?", (now, limit))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "event": json.loads(r[1]),
            "url": r[2],
            "attempts": r[3],
            "max_attempts": r[4],
            "base_backoff": r[5],
        })
    return out


def record_success(path: str | Path, id: int) -> None:
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute("DELETE FROM queue WHERE id=?", (id,))
    conn.commit()
    conn.close()


def record_failure(path: str | Path, id: int) -> None:
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    # increment attempts and set next_attempt
    cur.execute("SELECT attempts, base_backoff, max_attempts FROM queue WHERE id=?", (id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return
    attempts, base_backoff, max_attempts = row
    attempts += 1
    # exponential backoff
    next_attempt = time.time() + base_backoff * (2 ** (attempts - 1))
    if attempts >= max_attempts:
        # move to dead_letter
        cur.execute("SELECT event, url FROM queue WHERE id=?", (id,))
        ev, url = cur.fetchone()
        cur.execute("INSERT INTO dead_letter (event, url, attempts, failed_at) VALUES (?, ?, ?, strftime('%s','now'))", (ev, url, attempts))
        cur.execute("DELETE FROM queue WHERE id=?", (id,))
    else:
        cur.execute("UPDATE queue SET attempts=?, next_attempt=? WHERE id=?", (attempts, next_attempt, id))
    conn.commit()
    conn.close()


def clear_db(path: str | Path) -> None:
    if Path(path).exists():
        Path(path).unlink()
