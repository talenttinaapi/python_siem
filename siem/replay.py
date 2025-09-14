"""Replay queued failed webhook events stored in a SQLite-backed queue."""
from pathlib import Path
import time
import argparse
from typing import Optional

from . import queue as _queue
from .alerts import send_webhook_once


def process_queue(db_path: str | Path = None, max_items: int = 100, client=None, silent: bool = False):
    if db_path is None:
        db_path = _queue.DEFAULT_DB
    _queue.init_db(db_path)
    items = _queue.list_due(db_path, limit=max_items)
    if not items:
        if not silent:
            print("No due items to replay")
        return
    for it in items:
        id = it["id"]
        ev = it["event"]
        url = it["url"]
        attempts = it["attempts"]
        max_attempts = it["max_attempts"]
        base_backoff = it["base_backoff"]
        ok = False
        try:
            ok = send_webhook_once(ev, url, client=client)
        except Exception:
            ok = False
        if ok:
            _queue.record_success(db_path, id)
            if not silent:
                print(f"Delivered queued id={id} to {url}")
        else:
            # update attempts/next_attempt or move to dead_letter
            _queue.record_failure(db_path, id)
            if not silent:
                print(f"Failed id={id}, attempts updated")


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(prog="siem-replay", description="Replay queued webhook events")
    parser.add_argument("--db", help="Path to queue DB", default=_queue.DEFAULT_DB)
    parser.add_argument("--max-items", type=int, default=100, help="Max items to process per run")
    parser.add_argument("--silent", action="store_true", help="Suppress output")
    args = parser.parse_args(argv)
    process_queue(db_path=args.db, max_items=args.max_items, silent=args.silent)


if __name__ == "__main__":
    main()
