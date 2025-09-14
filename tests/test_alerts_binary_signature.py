import unittest
from siem.alerts import sign_payload
import hmac
import hashlib


class TestBinarySignature(unittest.TestCase):
    def test_sign_bytes_and_str_compat(self):
        payload = b'{"msg":"hello"}'
        secret = 's3cr3t'
        # compute expected using hmac directly
        expected = hmac.new(secret.encode('utf-8'), payload, hashlib.sha256).hexdigest()
        got = sign_payload(payload, secret)
        self.assertEqual(expected, got)

    def test_sign_str(self):
        payload = '{"msg":"hello"}'
        secret = 's3cr3t'
        expected = hmac.new(secret.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        got = sign_payload(payload, secret)
        self.assertEqual(expected, got)


if __name__ == '__main__':
    unittest.main()
