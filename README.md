# Python SIEM System Prototype

This is a prototype SIEM (Security Information and Event Management) system written in Python.

##Features
- Log ingestion (batch + stream tail)
- Pluggable parsers (JSON, key=value, Apache common log)
- Pluggable parsers (JSON, key=value, Apache common log, CSV, syslog)
- TF-IDF + numeric feature extraction
- Anomaly detection (Isolation Forest, optional Autoencoder)
- LLM contextualization (OpenAI integration)
- Simple alerting system

## Installation

```bash
pip install -r requirements.txt
```

YAML config support: the app will load `config.json` by default but will also
accept YAML files (requires `pyyaml` to be installed).

## Usage

Quick run (process `logs/sample.log`):
```bash
python run.py
```

Tail a log file for real-time alerts:
```bash
python run.py logs/sample.log --watch
```

You can also run the package module directly:
```bash
python -m run logs/sample.log

After packaging you can also install an entry point `siem-run` via `pip`:

```bash
pip install .
siem-run logs/sample.log
```

Signed webhook verification example (receiver side):
```python
import hmac, hashlib

def verify(payload_bytes: bytes, secret: str, signature_hex: str) -> bool:
	expected = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
	return hmac.compare_digest(expected, signature_hex)

# On receiving a POST, read request.data and compare header X-SIEM-Signature
```

There is an example Flask receiver in `examples/webhook_receiver.py` demonstrating this.

Retry/backoff and error queue:
- Webhook deliveries use a simple retry with exponential backoff. Failed events are written to `logs/failed_webhooks.jsonl` for later replay.
```

## Replay and Dead-Letter Management

A persistent SQLite-backed queue is used to store failed webhook deliveries in `logs/failed_webhooks.db`.

Replay queued items (processes due items and respects per-item backoff and max attempts):

```bash
# Run the module directly
python -m siem.replay --db logs/failed_webhooks.db --max-items 100

# Or use the console script after installing the package
siem-replay --db logs/failed_webhooks.db --max-items 100
```

Inspect or requeue dead-lettered items (moved there after exceeding max attempts):

```bash
# List dead-letter entries
siem-dead-letter --db logs/failed_webhooks.db --list --limit 50

# Requeue specific ids (space-separated)
siem-dead-letter --db logs/failed_webhooks.db --requeue 3 7 11

# Requeue all dead-letter items
siem-dead-letter --db logs/failed_webhooks.db --requeue
```

The `siem-dead-letter` tool will move selected dead-letter rows back into the queue (with attempts reset) so they can be retried by `siem-replay`.


## Notes

