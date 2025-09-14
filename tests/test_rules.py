import unittest
from siem.rules import rule_http_5xx, rule_failed_login


class TestRules(unittest.TestCase):
    def test_http_5xx(self):
        ev = {'status': '500'}
        self.assertTrue(rule_http_5xx(ev))

    def test_failed_login(self):
        ev = {'msg': 'User login failed for bob'}
        self.assertTrue(rule_failed_login(ev))


if __name__ == '__main__':
    unittest.main()
