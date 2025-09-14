"""Log ingestion helpers."""
from pathlib import Path
from typing import Iterator


def tail_file(path: Path) -> Iterator[str]:
    """Yield new lines appended to a file (simple tail -f).

    This is a naive implementation intended for small projects and demos.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        # go to end
        fh.seek(0, 2)
        while True:
            line = fh.readline()
            if not line:
                import time

                time.sleep(0.2)
                continue
            yield line.rstrip("\n")


def read_file(path: Path) -> Iterator[str]:
    """Yield lines from a file.

    Useful for initial ingest of historical logs.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            yield line.rstrip("\n")
