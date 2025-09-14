"""Rule engine for detecting events of interest."""
from typing import Dict, Callable, List


Rule = Callable[[Dict], bool]


def rule_failed_login(event: Dict) -> bool:
    """Detects failed login messages by level or message content."""
    msg = (event.get("msg") or "").lower()
    level = (event.get("level") or "").lower()
    return "failed" in msg or "failed" in level


DEFAULT_RULES: List[Rule] = [rule_failed_login]


def rule_http_5xx(event: Dict) -> bool:
    status = event.get("status")
    try:
        if status and int(status) >= 500:
            return True
    except Exception:
        pass
    return False


DEFAULT_RULES: List[Rule] = [rule_failed_login, rule_http_5xx]
