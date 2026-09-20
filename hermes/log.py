"""Machine-readable JSONL logging plus a terse human stream on stderr.

Every record also carries the work that produced it -- which run, phase,
ticket and attempt were in scope. That scope is ambient rather than an
argument to each call, because the lines worth correlating are emitted deep
inside model, sandbox and verification code that has no business knowing which
ticket it serves. The loop sets it once; everything under it inherits.
"""
from __future__ import annotations

import contextvars
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}
_lock = threading.Lock()

#: The work in scope, stamped onto every record. A context variable and not a
#: Logger attribute: subsystems build their own loggers, and all of them belong
#: to the same run.
_scope: contextvars.ContextVar[dict] = contextvars.ContextVar("hermes_scope", default={})

#: Kept out of the human stream: constant for the whole process, so on stderr
#: it is noise. It is the field that makes a saved JSONL file readable later.
_QUIET_FIELDS = ("run",)


@contextmanager
def scope(**fields: Any) -> Iterator[dict]:
    """Stamp *fields* onto every record logged inside the block.

    Fields merge onto the enclosing scope, so an inner block naming *ticket*
    keeps the *run* set outside it. A ``None`` is dropped rather than logged.
    """
    merged = {**_scope.get(), **{k: v for k, v in fields.items() if v is not None}}
    token = _scope.set(merged)
    try:
        yield merged
    finally:
        _scope.reset(token)


def current_scope() -> dict:
    """The scope in force, to hand to a worker thread.

    Context variables do not cross a thread boundary, so work submitted to a
    pool would otherwise log with no run or ticket on it at all.
    """
    return dict(_scope.get())


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
            **_scope.get(),
            **fields,   # an explicit field wins: the caller knows better
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
            if k not in ("ts", "iso", "level", "event") + _QUIET_FIELDS
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
