import unittest
from unittest.mock import patch, MagicMock
from siem.alerts import webhook_alert


class TestWebhookAlert(unittest.TestCase):
    @patch('siem.alerts.requests.post')
    def test_webhook_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp
        # Should not raise
        webhook_alert({'msg': 'x'}, 'http://example.local/hook', logger=None)


if __name__ == '__main__':
    unittest.main()
