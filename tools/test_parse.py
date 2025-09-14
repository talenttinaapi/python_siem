from pathlib import Path
from siem.parse import parse_line
from siem.rules import DEFAULT_RULES


def main():
    p = Path("logs/sample.log")
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = parse_line(line)
        print("LINE:", line)
        print("PARSED:", ev)
        for r in DEFAULT_RULES:
            try:
                if r(ev):
                    print("RULE FIRED:", r.__name__)
            except Exception as e:
                print("RULE ERROR:", r.__name__, e)
        print()


if __name__ == '__main__':
    main()
