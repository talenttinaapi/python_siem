"""Example webhook receiver demonstrating signature verification.

Run with: `pip install flask` then `python examples/webhook_receiver.py`
"""
from flask import Flask, request, abort
import hmac
import hashlib

app = Flask(__name__)

# Shared secret (in production store securely)
SHARED_SECRET = "mysecret"


def verify_signature(payload_bytes: bytes, secret: str, signature_hex: str) -> bool:
    expected = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_hex)


@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-SIEM-Signature") or request.headers.get("X-Siem-Signature")
    if not signature:
        abort(400, "Missing signature")
    if not verify_signature(request.data, SHARED_SECRET, signature):
        abort(403, "Invalid signature")
    # process payload
    print("Received payload:", request.json)
    return ("OK", 200)


if __name__ == "__main__":
    app.run(port=8080)
