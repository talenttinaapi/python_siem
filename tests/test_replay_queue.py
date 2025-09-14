import os
import tempfile
import json
import time

from siem import queue
from siem.replay import process_queue


class FlakyClient:
    """Client that fails the first call and succeeds afterwards."""

    def __init__(self):
        self.calls = 0

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls += 1
        class Resp:
            def raise_for_status(self):
                return None

        if self.calls == 1:
            raise RuntimeError("simulated network error")
        return Resp()


def test_queue_and_replay_tmp_db(tmp_path):
    db = tmp_path / "queue.db"
    # init db
    queue.init_db(db)
    ev = {"msg": "test"}
    url = "http://example.local/webhook"
    # enqueue
    id = queue.enqueue(db, ev, url, max_attempts=2, base_backoff=0.1)
    items = queue.list_due(db)
    assert len(items) == 1

    client = FlakyClient()
    # first run: client will fail, so attempts should increment
    process_queue(db_path=db, max_items=10, client=client, silent=True)
    items = queue.list_due(db)
    # after failure (attempts=1), item should remain
    assert len(items) == 1
    assert items[0]["attempts"] == 1

    # second run: client will succeed and item should be removed
    process_queue(db_path=db, max_items=10, client=client, silent=True)
    items = queue.list_due(db)
    assert len(items) == 0

    # enqueue an item that will exceed attempts
    id2 = queue.enqueue(db, ev, url, max_attempts=1, base_backoff=0.1)
    # client fails always
    class AlwaysFail:
        def post(self, *a, **k):
            raise RuntimeError("boom")

    process_queue(db_path=db, max_items=10, client=AlwaysFail(), silent=True)
    # should be moved to dead_letter (no longer in due list)
    items = queue.list_due(db)
    assert all(i["id"] != id2 for i in items)
