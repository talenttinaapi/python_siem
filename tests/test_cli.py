import unittest
import logging
from siem.cli import process_lines
from siem.alerts import console_alert


class DummyHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def sample_rule(ev):
    return '500' in (ev.get('status') or '')


class TestCLI(unittest.TestCase):
    def test_process_lines_triggers_alert(self):
        logger = logging.getLogger('cli_test')
        handler = DummyHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        lines = [
            '127.0.0.1 - - [12/Sep/2025:06:26:11 +0000] "POST /login HTTP/1.1" 500 512'
        ]
        # pass a console alert bound to the test logger
        process_lines(lines, rules=[sample_rule], alerts=[lambda ev: console_alert(ev, logger=logger)], logger=logger)
        self.assertTrue(any('ALERT' in r.getMessage() for r in handler.records))


if __name__ == '__main__':
    unittest.main()
