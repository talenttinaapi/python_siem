import unittest
from siem.parse import parse_line


class TestMalformedParse(unittest.TestCase):
    def test_malformed_json_fallback(self):
        # a malformed JSON line should not raise, parse_line should try fallbacks
        line = '{"msg": "incomplete"'
        parsed = parse_line(line)
        # parsed should be a dict or None; assert it doesn't raise and returns something
        self.assertTrue(parsed is None or isinstance(parsed, dict))

    def test_binary_nonutf8_line(self):
        # simulate a binary blob that is not utf-8
        raw = b"\x80\x81\x82"
        try:
            s = raw.decode('utf-8')
        except Exception:
            s = None
        # feed fallback string or ensure parse_line handles bytes as well
        if s is None:
            # ensure passing a repr doesn't crash
            parsed = parse_line(str(raw))
        else:
            parsed = parse_line(s)
        self.assertTrue(parsed is None or isinstance(parsed, dict))


if __name__ == '__main__':
    unittest.main()
