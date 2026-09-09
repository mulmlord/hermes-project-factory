"""Machine-readable JSONL logging plus a terse human stream on stderr."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}
_lock = threading.Lock()


class Logger:
    def __init__(self, path: Path | None = None, level: str = "info", quiet: bool = False):
        self.path = Path(path) if path else None
        self.level = LEVELS.get(level, 20)
        self.quiet = quiet
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, level: str, event: str, **fields: Any) -> None:
        if LEVELS.get(level, 20) < self.level:
            return
        record = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "level": level,
            "event": event,
            **fields,
        }
        line = json.dumps(record, default=str)
        with _lock:
            if self.path:
                with self.path.open("a") as fh:
                    fh.write(line + "\n")
            if not self.quiet:
                sys.stderr.write(self._human(record) + "\n")
                sys.stderr.flush()

    @staticmethod
    def _human(record: dict) -> str:
        detail = " ".join(
            f"{k}={_short(v)}"
            for k, v in record.items()
            if k not in ("ts", "iso", "level", "event")
        )
        mark = {"debug": "  ", "info": "->", "warn": " !", "error": "XX"}[record["level"]]
        return f"{record['iso']} {mark} {record['event']}" + (f"  {detail}" if detail else "")

    def debug(self, event: str, **f: Any) -> None: self.log("debug", event, **f)
    def info(self, event: str, **f: Any) -> None: self.log("info", event, **f)
    def warn(self, event: str, **f: Any) -> None: self.log("warn", event, **f)
    def error(self, event: str, **f: Any) -> None: self.log("error", event, **f)


def _short(value: Any, limit: int = 120) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


NULL = Logger(None, level="error", quiet=True)


def for_workspace(root: Path, level: str | None = None, quiet: bool = False) -> Logger:
    stamp = time.strftime("%Y%m%d")
    return Logger(
        Path(root) / ".hermes" / "logs" / f"hermes-{stamp}.jsonl",
        level=level or os.environ.get("HERMES_LOG_LEVEL", "info"),
        quiet=quiet,
    )
