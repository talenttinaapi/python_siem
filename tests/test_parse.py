import unittest
from siem.parse import parse_line


class TestParse(unittest.TestCase):
    def test_apache_parsing(self):
        line = '127.0.0.1 - - [12/Sep/2025:06:26:11 +0000] "POST /login HTTP/1.1" 500 512'
        ev = parse_line(line)
        self.assertEqual(ev.get('host'), '127.0.0.1')
        self.assertEqual(ev.get('status'), '500')

    def test_fallback(self):
        line = '2025-09-12 INFO This is a test message'
        ev = parse_line(line)
        self.assertIn('msg', ev)

    def test_json_line(self):
        line = '{"timestamp": "2025-09-12T00:00:00Z", "level": "INFO", "msg": "ok"}'
        ev = parse_line(line)
        self.assertEqual(ev.get('level'), 'INFO')

    def test_kv_line(self):
        line = 'user=bob action=login status=success'
        ev = parse_line(line)
        self.assertEqual(ev.get('user'), 'bob')
        self.assertEqual(ev.get('status'), 'success')

    def test_csv_line(self):
        line = 'one,two,three'
        ev = parse_line(line)
        self.assertEqual(ev.get('col0'), 'one')
        self.assertEqual(ev.get('col2'), 'three')

    def test_syslog_line(self):
        line = 'Jul 27 12:34:56 myhost myprogram: something happened'
        ev = parse_line(line)
        self.assertEqual(ev.get('host'), 'myhost')
        self.assertIn('timestamp', ev)


if __name__ == '__main__':
    unittest.main()
