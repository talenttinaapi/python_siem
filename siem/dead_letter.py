"""Inspect and manage dead-lettered webhook events."""
import argparse
from pathlib import Path
from typing import Optional

from . import queue as _queue


def list_dead(db_path: str | Path = None, limit: int = 100):
    if db_path is None:
        db_path = _queue.DEFAULT_DB
    _queue.init_db(db_path)
    conn = __import__('sqlite3').connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT id, event, url, attempts, failed_at FROM dead_letter ORDER BY failed_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            'id': r[0],
            'event': r[1],
            'url': r[2],
            'attempts': r[3],
            'failed_at': r[4],
        })
    return out


def requeue(db_path: str | Path = None, ids: list[int] | None = None):
    if db_path is None:
        db_path = _queue.DEFAULT_DB
    _queue.init_db(db_path)
    conn = __import__('sqlite3').connect(str(db_path))
    cur = conn.cursor()
    if ids is None:
        cur.execute("SELECT id, event, url, attempts FROM dead_letter")
        rows = cur.fetchall()
    else:
        rows = []
        for i in ids:
            cur.execute("SELECT id, event, url, attempts FROM dead_letter WHERE id=?", (i,))
            r = cur.fetchone()
            if r:
                rows.append(r)
    for r in rows:
        _id, ev, url, attempts = r
        try:
            obj = ev
            # insert back into queue with remaining attempts reset
            _queue.enqueue(db_path, __import__('json').loads(obj) if isinstance(obj, str) else obj, url, max_attempts=max(1, attempts))
            cur.execute("DELETE FROM dead_letter WHERE id=?", (_id,))
        except Exception:
            continue
    conn.commit()
    conn.close()


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(prog="siem-dead-letter", description="Inspect and manage dead-letter queue")
    parser.add_argument("--db", default=_queue.DEFAULT_DB, help="Path to queue DB")
    parser.add_argument("--list", action="store_true", help="List dead-letter entries")
    parser.add_argument("--requeue", nargs="*", help="IDs to requeue (space separated). Use no args to requeue all.")
    parser.add_argument("--limit", type=int, default=100, help="Limit list output")
    args = parser.parse_args(argv)
    if args.list:
        rows = list_dead(args.db, limit=args.limit)
        for r in rows:
            print(f"id={r['id']} url={r['url']} attempts={r['attempts']} failed_at={r['failed_at']} event={r['event']}")
        return
    if args.requeue is not None:
        ids = None
        if len(args.requeue) > 0:
            ids = [int(i) for i in args.requeue]
        requeue(args.db, ids=ids)
        print("Requeue requested")


if __name__ == '__main__':
    main()
