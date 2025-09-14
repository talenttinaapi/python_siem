import tempfile
from siem import queue
from siem import dead_letter_admin as admin
import json


def test_export_import_roundtrip(tmp_path):
    db = tmp_path / "queue.db"
    queue.init_db(db)
    ev = {"msg": "dlx"}
    url = "http://example.local/hook"
    # insert into dead_letter directly
    conn = __import__('sqlite3').connect(str(db))
    cur = conn.cursor()
    cur.execute("INSERT INTO dead_letter (event, url, attempts, failed_at) VALUES (?, ?, ?, ?)", (json.dumps(ev), url, 2, __import__('time').time()))
    conn.commit()
    conn.close()

    out = tmp_path / "export.jsonl"
    admin.export_dead(db_path=db, out_file=out)
    assert out.exists()
    # import into a new db and requeue
    db2 = tmp_path / "queue2.db"
    admin.import_dead(db_path=db2, in_file=out, requeue=True)
    # ensure queue has an item
    items = queue.list_due(db2)
    assert len(items) >= 1
