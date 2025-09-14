"""Configuration and logging helpers."""
import logging
import json
from typing import Dict, Callable, Any
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=lvl, format=fmt)
    return logging.getLogger("siem")


def rule_registry(rules: Dict[str, Callable]) -> Dict[str, Callable]:
    """Return the passed mapping; small helper for future expansion."""
    return rules


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        # prefer JSON for zero-dependency
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            return {}


def get_enabled_rules_config(cfg: Dict[str, Any]) -> list:
    """Return list of rule names enabled via config (or empty)."""
    return cfg.get("rules", []) or []


def get_alerts_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("alerts", {})
