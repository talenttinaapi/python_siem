import unittest
from unittest.mock import patch, MagicMock
from siem.alerts import webhook_alert, sign_payload


class TestWebhookSignature(unittest.TestCase):
    @patch('siem.alerts.requests.post')
    def test_signature_added(self, mock_post):
        captured = {}

        def fake_post(url, data=None, headers=None, timeout=None):
            captured['url'] = url
            captured['headers'] = headers
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status.return_value = None
            return mock_resp

        mock_post.side_effect = fake_post
        payload = {'msg': 'hello'}
        secret = 'mysecret'
        webhook_alert(payload, 'http://example.local/hook', secret=secret, logger=None)
        headers = {k.lower(): v for k, v in (captured.get('headers') or {}).items()}
        self.assertIn('x-siem-signature', headers)
        # validate signature matches payload
        import json

        expected = sign_payload(json.dumps(payload).encode('utf-8'), secret)
        self.assertEqual(headers['x-siem-signature'], expected)


if __name__ == '__main__':
    unittest.main()
