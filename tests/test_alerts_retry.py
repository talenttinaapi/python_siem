import unittest
from unittest.mock import patch
from siem.alerts import RequestException
from pathlib import Path
from siem.alerts import webhook_alert


class TestWebhookRetry(unittest.TestCase):
    @patch('siem.alerts.requests.post', side_effect=RequestException('fail'))
    def test_retry_and_queue(self, mock_post):
        q = Path('logs/failed_webhooks.jsonl')
        if q.exists():
            q.unlink()
        webhook_alert({'msg': 'will fail'}, 'http://example.local/hook', secret=None, logger=None)
        self.assertTrue(q.exists())
        content = q.read_text(encoding='utf-8')
        self.assertIn('will fail', content)


if __name__ == '__main__':
    unittest.main()
