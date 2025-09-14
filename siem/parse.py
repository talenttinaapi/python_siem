"""Parsing utilities for log lines."""
import re
import json
from typing import Dict, Optional


LOG_PATTERN = re.compile(r"(?P<timestamp>\S+)\s+(?P<level>\S+)\s+(?P<msg>.*)")

# Apache common log simple pattern: remote host - - [date] "METHOD path HTTP/x" status bytes
APACHE_PATTERN = re.compile(r"(?P<host>\S+)\s+\S+\s+\S+\s+\[(?P<timestamp>[^\]]+)\]\s+\"(?P<request>[^\"]+)\"\s+(?P<status>\d{3})\s+(?P<size>\d+)")


def parse_json_line(line: str) -> Optional[Dict]:
    """Attempt to parse a JSON-formatted log line."""
    try:
        return json.loads(line)
    except Exception:
        return None


def parse_kv_line(line: str) -> Optional[Dict]:
    """Parse key=value pairs into a dict (e.g. 'a=1 b=two')."""
    parts = line.split()
    out: Dict[str, str] = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k] = v
    return out if out else None


def parse_csv_line(line: str) -> Optional[Dict]:
    """Parse a simple CSV line into a dict with indexed keys if headerless.

    If the line contains commas and looks like CSV, return a mapping like
    {'col0': value0, 'col1': value1, ...}.
    """
    if "," not in line:
        return None
    parts = [p.strip() for p in line.split(",")]
    return {f"col{i}": v for i, v in enumerate(parts)}


SYSLOG_PATTERN = re.compile(r"(?P<month>\w{3})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<host>\S+)\s+(?P<msg>.+)")


def parse_syslog_line(line: str) -> Optional[Dict]:
    """Parse a BSD syslog-style line like 'Jul 27 12:34:56 host program: message'."""
    m = SYSLOG_PATTERN.match(line)
    if not m:
        return None
    gd = m.groupdict()
    gd['timestamp'] = f"{gd.pop('month')} {gd.pop('day')} {gd.pop('time')}"
    return gd


def parse_line(line: str) -> Dict[str, Optional[str]]:
    """Parse a log line into structured fields.

    Detection order: JSON -> key=value -> Apache -> generic pattern -> fallback raw message.
    """
    j = parse_json_line(line)
    if j is not None:
        return j

    kv = parse_kv_line(line)
    if kv is not None:
        return kv

    csv = parse_csv_line(line)
    if csv is not None:
        return csv

    syslog = parse_syslog_line(line)
    if syslog is not None:
        return syslog

    m = APACHE_PATTERN.match(line)
    if m:
        gd = m.groupdict()
        gd.setdefault("level", None)
        gd.setdefault("msg", gd.get("request"))
        return gd

    m = LOG_PATTERN.match(line)
    if not m:
        return {"timestamp": None, "level": None, "msg": line}
    return m.groupdict()
