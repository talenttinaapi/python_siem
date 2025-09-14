"""Initialize the SIEM runtime SQLite DB used for failed webhook retries.

This script creates `logs/failed_webhooks.db` with the same schema used by
`siem.queue.init_db`. It is safe to run multiple times.
"""
import sqlite3
from pathlib import Path


DEFAULT_DB = Path("logs/failed_webhooks.db")


def init_db(path: Path | str = DEFAULT_DB) -> None:
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


def main() -> None:
    init_db()
    print(f"Initialized DB at: {DEFAULT_DB}")


if __name__ == "__main__":
    main()
