import unittest
from siem.config import load_config, get_enabled_rules_config
from pathlib import Path


class TestConfig(unittest.TestCase):
    def test_load_json(self):
        cfg = load_config(Path('config.json'))
        self.assertIsInstance(cfg, dict)
        rules = get_enabled_rules_config(cfg)
        self.assertIn('rule_http_5xx', rules)


if __name__ == '__main__':
    unittest.main()
