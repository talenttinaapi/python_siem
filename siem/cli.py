"""CLI helpers for running the SIEM system."""
from pathlib import Path
from typing import Iterable, Callable, Dict, List
import logging

from .ingest import read_file, tail_file
from .parse import parse_line
from .alerts import console_alert


def process_lines(
    lines: Iterable[str],
    rules: Iterable[Callable[[dict], bool]],
    alerts: List[Callable[[dict], None]] | None = None,
    logger: logging.Logger | None = None,
):
    alerts = alerts or []
    for line in lines:
        event = parse_line(line)
        for rule in rules:
            try:
                if rule(event):
                    for a in alerts:
                        try:
                            a(event)
                        except Exception:
                            if logger:
                                logger.exception("Alert handler error: %s", getattr(a, "__name__", str(a)))
            except Exception:
                if logger:
                    logger.exception("Rule error: %s", getattr(rule, "__name__", str(rule)))


def run_watch(path: Path, rules: Iterable[Callable], alerts: List[Callable] | None = None, logger: logging.Logger | None = None):
    process_lines(tail_file(path), rules, alerts=alerts, logger=logger)


def run_once(path: Path, rules: Iterable[Callable], alerts: List[Callable] | None = None, logger: logging.Logger | None = None):
    process_lines(read_file(path), rules, alerts=alerts, logger=logger)
