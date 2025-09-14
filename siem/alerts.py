"""Alerting helpers (console and hooks)."""
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable
import time
import json
import hmac
import hashlib
import logging

from . import queue as _queue

try:
    import requests as requests
    from requests.exceptions import RequestException
except Exception:  # pragma: no cover - environments without requests
    class RequestException(Exception):
        pass

    class _RequestsShim:
        @staticmethod
        def post(u, data=None, headers=None, timeout=None):
            raise RequestException("No HTTP client available")

    requests = _RequestsShim()


def sign_payload(payload: Dict[str, Any], secret: str) -> str:
    # accept bytes/str/dict for backwards compatibility with tests
    if isinstance(payload, (bytes, bytearray)):
        body_bytes = bytes(payload)
    elif isinstance(payload, str):
        body_bytes = payload.encode("utf-8")
    else:
        # use default json.dumps() to match existing tests that compute
        # expected = sign_payload(json.dumps(payload).encode('utf-8'), secret)
        body_bytes = json.dumps(payload).encode("utf-8")
    mac = hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256)
    return mac.hexdigest()


def verify_signature(payload: Dict[str, Any], secret: str, signature: str) -> bool:
    expected = sign_payload(payload, secret)
    return hmac.compare_digest(expected, signature)


def console_alert(event: Dict[str, Any], logger: Optional[logging.Logger] = None):
    if logger is None:
        logger = logging.getLogger("siem.alerts")
    logger.warning("ALERT: %s", json.dumps(event))


def send_webhook_once(event: Dict[str, Any], url: str, secret: Optional[str] = None, client=None, timeout: int = 5) -> bool:
    """Attempt to send a single webhook. Returns True on success, False on permanent failure.

    `client` is expected to implement a `post(url, data, headers, timeout)` and raise on network errors.
    This keeps the sending logic testable and pluggable.
    """
    body = json.dumps(event)
    headers = {"Content-Type": "application/json"}
    if secret is not None:
        headers["X-SIEM-Signature"] = sign_payload(event, secret)
    if client is None:
        # use module-level requests (so tests can patch `siem.alerts.requests.post`)
        client = requests
        if client is None:
            # provide a minimal shim that mimics requests.post for environments without requests
            class _Shim:
                @staticmethod
                def post(u, data=None, headers=None, timeout=None):
                    raise RequestException("No HTTP client available")

            client = _Shim()
    resp = None
    try:
        resp = client.post(url, data=body, headers=headers, timeout=timeout)
        # requests-like response
        if hasattr(resp, "raise_for_status"):
            resp.raise_for_status()
            return True
        # urllib-like: status/code
        if hasattr(resp, "status"):
            if 200 <= resp.status < 300:
                return True
        return False
    except Exception:
        return False


def webhook_alert(event: Dict[str, Any], url: str, secret: Optional[str] = None, logger: Optional[logging.Logger] = None, db_path: str | Path = None, max_attempts: int = 3, base_backoff: float = 0.5, client=None) -> bool:
    """High-level webhook alert: try once, enqueue on failure.

    If `db_path` is provided, it will be used to store retries in a persistent SQLite DB.
    """
    if logger is None:
        logger = logging.getLogger("siem.alerts")
    ok = send_webhook_once(event, url, secret=secret, client=client)
    if ok:
        logger.info("webhook delivered %s", url)
        return True
    logger.warning("webhook delivery failed, enqueueing for retry: %s", url)
    # enqueue to sqlite DB
    if db_path is None:
        db_path = _queue.DEFAULT_DB
    _queue.init_db(db_path)
    _queue.enqueue(db_path, event, url, max_attempts=max_attempts, base_backoff=base_backoff)
    # also write legacy JSONL queue for backward compatibility/tests
    try:
        q = Path("logs/failed_webhooks.jsonl")
        q.parent.mkdir(parents=True, exist_ok=True)
        with q.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": event, "url": url}) + "\n")
    except Exception:
        pass
    return False


def build_alert_callables(cfg: dict, logger: logging.Logger | None = None) -> List[Callable[[Dict], None]]:
    """Given a config dict, return a list of alert callables.

    Supports `console: true` and `webhook: <url>` in `cfg['alerts']`.
    """
    out: List[Callable[[Dict], None]] = []
    alerts = cfg.get("alerts", {}) if cfg else {}
    if alerts.get("console"):
        out.append(lambda ev: console_alert(ev, logger=logger))
    webhook = alerts.get("webhook")
    if webhook:
        secret = alerts.get("webhook_secret")
        max_attempts = alerts.get("webhook_max_attempts") or 3
        base_backoff = alerts.get("webhook_base_backoff") or 0.5
        out.append(
            lambda ev: webhook_alert(
                ev, webhook, secret, logger=logger, max_attempts=max_attempts, base_backoff=base_backoff
            )
        )
    return out
