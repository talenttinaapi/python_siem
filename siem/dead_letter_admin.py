"""Admin utilities for exporting and importing dead-letter items."""
import argparse
from pathlib import Path
from typing import Optional
import json

from . import queue as _queue


def export_dead(db_path: str | Path = None, out_file: str | Path = "dead_letter_export.jsonl"):
    if db_path is None:
        db_path = _queue.DEFAULT_DB
    _queue.init_db(db_path)
    conn = __import__('sqlite3').connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT id, event, url, attempts, failed_at FROM dead_letter ORDER BY failed_at")
    rows = cur.fetchall()
    conn.close()
    p = Path(out_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"id": r[0], "event": r[1], "url": r[2], "attempts": r[3], "failed_at": r[4]}) + "\n")


def import_dead(db_path: str | Path = None, in_file: str | Path = "dead_letter_export.jsonl", requeue: bool = False):
    if db_path is None:
        db_path = _queue.DEFAULT_DB
    _queue.init_db(db_path)
    p = Path(in_file)
    if not p.exists():
        raise FileNotFoundError(in_file)
    conn = __import__('sqlite3').connect(str(db_path))
    cur = conn.cursor()
    with p.open("r", encoding="utf-8") as fh:
        for ln in fh:
            obj = json.loads(ln)
            ev = obj.get('event')
            url = obj.get('url')
            attempts = obj.get('attempts') or 0
            # Insert into dead_letter table
            cur.execute("INSERT INTO dead_letter (event, url, attempts, failed_at) VALUES (?, ?, ?, ?)", (json.dumps(ev) if not isinstance(ev, str) else ev, url, attempts, obj.get('failed_at') or __import__('time').time()))
            if requeue:
                # re-enqueue immediately
                _queue.enqueue(db_path, ev, url, max_attempts=max(1, attempts))
    conn.commit()
    conn.close()


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(prog='siem-dead-letter-admin', description='Export/import dead-letter entries')
    sub = parser.add_subparsers(dest='cmd')
    ex = sub.add_parser('export')
    ex.add_argument('--db', default=_queue.DEFAULT_DB)
    ex.add_argument('--out', default='dead_letter_export.jsonl')
    im = sub.add_parser('import')
    im.add_argument('--db', default=_queue.DEFAULT_DB)
    im.add_argument('--in', dest='infile', required=True)
    im.add_argument('--requeue', action='store_true')
    args = parser.parse_args(argv)
    if args.cmd == 'export':
        export_dead(args.db, out_file=args.out)
        print('Exported')
    elif args.cmd == 'import':
        import_dead(args.db, in_file=args.infile, requeue=args.requeue)
        print('Imported')


if __name__ == '__main__':
    main()
