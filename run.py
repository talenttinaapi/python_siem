"""Entrypoint to run the SIEM demo package."""
from pathlib import Path
import argparse

from siem.cli import run_once, run_watch
from siem import rules as rules_module
from siem.config import setup_logging, rule_registry, load_config, get_enabled_rules_config
from siem.alerts import build_alert_callables
import logging
from typing import List


def main():
    p = argparse.ArgumentParser(description="Run simple SIEM demo")
    p.add_argument("path", nargs="?", default="logs/sample.log")
    p.add_argument("--watch", action="store_true", help="Tail the file")
    p.add_argument(
        "--log-level",
        default=None,
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    p.add_argument("--config", default="config.json", help="Path to config JSON/YAML")
    p.add_argument(
        "--rules",
        nargs="*",
        help="List of rule function names to enable (defaults to all)",
    )
    args = p.parse_args()
    path = Path(args.path)
    cfg = load_config(args.config)
    # CLI flag overrides config file
    chosen_level = args.log_level or cfg.get("log_level", "INFO")
    logger = setup_logging(chosen_level)

    available = {name: getattr(rules_module, name) for name in dir(rules_module) if callable(getattr(rules_module, name))}
    enabled: List = []
    if args.rules:
        for r in args.rules:
            if r in available:
                enabled.append(available[r])
            else:
                logger.warning("Unknown rule requested: %s", r)
    else:
        # check config for rules
        cfg_rules = get_enabled_rules_config(cfg)
        if cfg_rules:
            for r in cfg_rules:
                if r in available:
                    enabled.append(available[r])
                else:
                    logger.warning("Unknown rule in config: %s", r)
        else:
            # use DEFAULT_RULES
            enabled = rules_module.DEFAULT_RULES

    enabled = rule_registry({n: f for n, f in available.items() if f in enabled}).values() if isinstance(enabled, list) else enabled

    alerts = build_alert_callables(cfg, logger=logger)

    if args.watch:
        run_watch(path, enabled, alerts=alerts, logger=logger)
    else:
        run_once(path, enabled, alerts=alerts, logger=logger)


if __name__ == "__main__":
    main()
