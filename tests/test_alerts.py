import unittest
import logging
from siem.alerts import console_alert


class DummyHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class TestAlerts(unittest.TestCase):
    def test_console_alert_logs(self):
        logger = logging.getLogger('test_alert')
        handler = DummyHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        console_alert({'msg': 'x'}, logger=logger)
        self.assertTrue(any('ALERT' in r.getMessage() for r in handler.records))


if __name__ == '__main__':
    unittest.main()
