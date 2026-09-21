"""Read-only workspace lookups an agent may make before it answers.

Every agent here is otherwise single-shot: it is handed a prompt and returns
JSON. What it is handed is a guess -- the ticket's artifact list was written
before any code existed, and the journal's hints name only what earlier
tickets wrote -- so the one thing an implementer cannot do is check a name it
is unsure of. A local coding model that cannot check does not say so; it
invents a plausible signature, and that costs a whole generate/verify/triage
cycle to discover.

Three read-only tools close that gap: :func:`read_file`, :func:`grep` and
:func:`list_dir`, under a fixed call budget, spoken in the same JSON the
agents already answer in. A text protocol rather than a function-calling API
because Hermes must work against any local OpenAI-compatible server, and only
some of them speak ``tools=``.

A fourth, :func:`recall`, reads the context database instead of the disk: it
answers "what did the others already find out about this" from what agents
recorded, which is the one question no amount of reading the tree can settle.
It is offered only where a database exists.

Nothing here writes, and every path goes through :meth:`Sandbox.resolve`,
which already refuses anything outside the workspace. A refusal comes back as
an observation the model can act on rather than an exception: telling it the
file is not there is the point of the tool, not a fault.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .log import Logger, NULL
from .sandbox import SandboxError

#: What a lookup may name, and the order the protocol lists them in.
TOOLS = ("read_file", "grep", "list_dir", "recall")

#: Keys a model may put its lookups under. The first is what the protocol
#: asks for; the rest are what models trained on other harnesses reach for.
REQUEST_KEYS = ("lookup", "lookups", "tool_calls", "tools")

#: Directories a search never walks: version control, build output, installed
#: packages and the factory's own state. None of them is code the ticket is
#: about, and all of them are large enough to swallow the whole budget.
SKIP = frozenset({".git", ".hermes", "node_modules", ".venv", "venv", "artifacts",
                  "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
                  ".tox", ".eggs"})


@dataclass
class Call:
    """One lookup and what it returned, in the form the model reads it back."""

    tool: str
    detail: str
    output: str
    ok: bool = True

    def render(self) -> str:
        return f"> {self.tool} {self.detail}\n{self.output.rstrip()}"


class Toolbox:
    """The read-only tools one generation may use, and what it has left.

    A toolbox belongs to a single generation: the budget is spent by the
    agent answering one prompt, and the next prompt gets a fresh one.
    """

    #: Characters one file read returns before it is cut short. A model that
    #: needs the rest asks again from the line it stopped at.
    FILE_CHARS = 6000
    #: Matching lines one search returns, and the width of each.
    MATCHES = 40
    LINE_CHARS = 300
    #: Entries one directory listing returns.
    ENTRIES = 120
    #: Results one recall returns, and the width of each. Deliberately tighter
    #: than a file read: these are pointers into work already recorded, and an
    #: agent that wants one of them whole can read the file it names.
    RECALLS = 6
    RECALL_CHARS = 700
    #: A file larger than this is data, not source, and is never searched.
    MAX_BYTES = 1_000_000
    #: Files one search walks before it gives up on the pattern.
    MAX_FILES = 4000
    #: Seconds one search may spend. A model is free to write a pattern that
    #: backtracks, and nothing else in the loop would notice it stop.
    SEARCH_SECONDS = 5.0

    def __init__(self, sandbox, *, budget: int = 6, rounds: int = 3,
                 db=None, log: Logger = NULL):
        self.sandbox = sandbox
        #: The context database, or None. Without one the toolbox is exactly
        #: the three filesystem lookups it was before.
        self.db = db
        self.budget = max(0, int(budget))
        self.rounds = max(0, int(rounds))
        self.log = log
        self.calls: list[Call] = []
        self.used_rounds = 0

    # -- budget ----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.budget > 0 and self.rounds > 0

    @property
    def available(self) -> tuple[str, ...]:
        """The tools this toolbox can actually serve."""
        return TOOLS if self.db is not None else tuple(
            name for name in TOOLS if name != "recall")

    @property
    def remaining(self) -> int:
        return max(0, self.budget - len(self.calls))

    @property
    def rounds_left(self) -> int:
        return max(0, self.rounds - self.used_rounds)

    @property
    def spent(self) -> bool:
        return not self.remaining or not self.rounds_left

    def round(self, requests: list[Any]) -> list[Call]:
        """Run one round of requested lookups, up to what the budget allows.

        A malformed request costs a call like any other. That is what bounds
        the loop: every round spends at least one, so a model that only ever
        asks nonsense still runs out and is made to answer.
        """
        self.used_rounds += 1
        made = []
        for request in requests:
            if not self.remaining:
                self.log.warn("lookup.truncated", dropped=len(requests) - len(made))
                break
            made.append(self.invoke(request))
        return made

    # -- dispatch --------------------------------------------------------
    def invoke(self, request: Any) -> Call:
        """Run one lookup. Every outcome is an observation, never a raise."""
        if not isinstance(request, dict):
            call = Call("lookup", str(request)[:80], ok=False, output=(
                'a lookup must be an object, for example '
                '{"tool": "read_file", "path": "src/thing.py"}'))
        else:
            name = str(request.get("tool") or request.get("name") or "").strip()
            handler = {"read_file": self._read_file, "grep": self._grep,
                       "list_dir": self._list_dir,
                       "recall": self._recall}.get(name)
            if handler is None or name not in self.available:
                call = Call(name or "lookup", "", ok=False, output=(
                    f"there is no {name!r} tool; the tools are "
                    + ", ".join(self.available)))
            else:
                try:
                    detail, output, ok = handler(request)
                except SandboxError as exc:
                    detail, ok = _path(request), False
                    output = (f"refused: {exc}. Only paths inside the project "
                              "are readable, and they are relative to its root.")
                except OSError as exc:
                    detail, ok = _path(request), False
                    output = f"could not be read: {exc}"
                call = Call(name, detail, output, ok)
        self.calls.append(call)
        self.log.info("lookup.call", tool=call.tool, detail=call.detail[:120],
                      ok=call.ok, chars=len(call.output), left=self.remaining)
        return call

    def _read_file(self, request: dict) -> tuple[str, str, bool]:
        path = _path(request)
        start = max(1, _int(request.get("start"), 1))
        detail = path if start == 1 else f"{path} from line {start}"
        target = self.sandbox.resolve(path)
        if target.is_dir():
            return detail, "that is a directory; use list_dir for it", False
        if not target.is_file():
            return detail, "no such file in the workspace", False
        lines = target.read_text(errors="replace").splitlines()
        if start > len(lines):
            return detail, f"the file has only {len(lines)} lines", False
        body = "\n".join(lines[start - 1:])
        last = len(lines)
        if len(body) > self.FILE_CHARS:
            body = body[: self.FILE_CHARS]
            last = start + max(0, body.count("\n") - 1)
            body += f"\n… cut off here; read on with start: {last + 1}"
        return detail, f"(lines {start}-{last} of {len(lines)})\n{body}", True

    def _grep(self, request: dict) -> tuple[str, str, bool]:
        pattern = str(request.get("pattern") or request.get("query") or "")
        where = _path(request, default=".")
        detail = f"{pattern} in {where}"
        if not pattern:
            return detail, 'no "pattern" given', False
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return detail, f"not a valid regular expression ({exc})", False
        root = self.sandbox.resolve(where)
        if not root.exists():
            return detail, "no such file or directory in the workspace", False
        hits: list[str] = []
        deadline = time.monotonic() + self.SEARCH_SECONDS
        for file in self._walk(root):
            if time.monotonic() > deadline:
                hits.append("… search gave up here; narrow the pattern or the path")
                break
            try:
                text = file.read_text(errors="replace")
            except OSError:
                continue
            if "\x00" in text[:4096]:
                continue
            rel = file.relative_to(self.sandbox.root)
            for number, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{rel}:{number}: {line.strip()[: self.LINE_CHARS]}")
                    if len(hits) >= self.MATCHES:
                        hits.append("… more matches; narrow the pattern")
                        return detail, "\n".join(hits), True
        return detail, "\n".join(hits) or "no matches", True

    def _list_dir(self, request: dict) -> tuple[str, str, bool]:
        where = _path(request, default=".")
        target = self.sandbox.resolve(where)
        if target.is_file():
            return where, "that is a file; use read_file for it", False
        if not target.is_dir():
            return where, "no such directory in the workspace", False
        entries = sorted((p for p in target.iterdir() if p.name not in SKIP),
                         key=lambda p: (p.is_file(), p.name))
        listing = [f"{p.name}/" if p.is_dir() else f"{p.name}  {_size(p)}"
                   for p in entries[: self.ENTRIES]]
        if len(entries) > self.ENTRIES:
            listing.append(f"… {len(entries) - self.ENTRIES} more")
        return where, "\n".join(listing) or "(empty)", True

    def _recall(self, request: dict) -> tuple[str, str, bool]:
        """What agents recorded about something, from the context database."""
        query = str(request.get("query") or request.get("pattern")
                    or request.get("q") or "").strip()
        where = _path(request)
        detail = query + (f" in {where}" if where else "")
        if self.db is None:
            return detail, "this project keeps no context database", False
        if not query:
            return detail, 'no "query" given', False
        hits = self.db.search(query, limit=self.RECALLS, path=where)
        if not hits:
            # A true answer, and a useful one: it means no earlier agent left
            # anything about this, so there is nothing to be consistent with.
            return detail, "nothing recorded about that", True
        return detail, "\n\n".join(h.render()[: self.RECALL_CHARS] for h in hits), True

    def _walk(self, root: Path):
        if root.is_file():
            yield root
            return
        seen = 0
        for path in sorted(root.rglob("*")):
            if seen >= self.MAX_FILES:
                return
            if any(part in SKIP for part in path.parts) or not path.is_file():
                continue
            try:
                if path.stat().st_size > self.MAX_BYTES:
                    continue
            except OSError:
                continue
            seen += 1
            yield path

    # -- what the model sees ---------------------------------------------
    def protocol(self) -> str:
        """The lookup format, and how much of it is left to spend."""
        return (
            "BEFORE YOU ANSWER you may look at the workspace. To look, reply with "
            'only this:\n'
            '{"lookup": [{"tool": "read_file", "path": "src/thing.py"}]}\n'
            "The tools are read-only and every path is relative to the project root:\n"
            '  {"tool": "read_file", "path": "src/thing.py", "start": 1}'
            '   the file, from line "start" (optional)\n'
            '  {"tool": "grep", "pattern": "def parse", "path": "src"}'
            '      matching lines; "path" optional\n'
            '  {"tool": "list_dir", "path": "src"}'
            '                          what one directory holds\n'
            + ('  {"tool": "recall", "query": "retry backoff"}'
               '                 what other agents recorded about it\n'
               if self.db is not None else "")
            + f"You have {self.remaining} lookup(s) left across {self.rounds_left} more "
            "repl(y/ies). The results come back and you are asked again.\n"
            "Look up anything you would otherwise guess: a function you are about to "
            "call, its arguments, an import path, a file you were not shown. A guessed "
            "name costs a whole failed build.\n"
            'When you have what you need, answer the task as asked, with no "lookup" key.'
        )

    def transcript(self, limit: int | None = None) -> str:
        """The most recent calls that fit in *limit* characters.

        Oldest first in the text, but dropped oldest first when they do not
        all fit: the last thing looked up is the thing being answered with.
        """
        kept: list[str] = []
        used = 0
        for call in reversed(self.calls):
            text = call.render()
            if limit is not None and used + len(text) + 2 > limit:
                break
            kept.append(text)
            used += len(text) + 2
        if not kept and self.calls:
            # Not even the newest fits whole. Its head still names the tool and
            # the path, which is more use than an empty heading.
            return self.calls[-1].render()[:limit]
        return "\n\n".join(reversed(kept))


class Lookup:
    """One agent's chance to read the workspace before it answers.

    Constructed after every other block of *prompt* has been added, so the
    protocol and the results are charged against the budget before any block
    that sizes itself against the room left. With no toolbox, or one whose
    budget is zero, this is exactly the single-shot ask it replaces.
    """

    HEADER = "WHAT YOU LOOKED UP:\n"
    SPENT = ("\n\nYour lookup budget is spent. Answer the task now with what you "
             'have; a further "lookup" will not be run.')

    def __init__(self, agent, prompt, toolbox: Toolbox | None = None,
                 log: Logger = NULL):
        self.agent = agent
        self.prompt = prompt
        self.toolbox = toolbox if toolbox is not None and toolbox.enabled else None
        self.log = log
        self.results = self.protocol = None
        if self.toolbox is not None:
            # Results outrank everything the caller sizes itself (sources, the
            # change under review): the model asked for these, and a prompt
            # that drops them asks the same question again. The protocol is
            # never trimmed -- half a protocol teaches the wrong syntax -- but
            # it can be dropped whole, leaving a plain single-shot prompt.
            self.results = prompt.add("lookup", priority=15, floor=600, tail=True)
            self.protocol = prompt.add("tools", self.toolbox.protocol(), priority=5)

    def ask(self, **kw: Any) -> Any:
        """Answer the prompt, serving any lookups asked for along the way."""
        if self.toolbox is None:
            return self.agent.ask_json(self.prompt.render(), **kw)
        while True:
            payload = self.agent.ask_json(self.prompt.render(), **kw)
            requests = requested(payload)
            if not requests or self.toolbox.spent:
                if self.toolbox.calls:
                    # Not answering is worth finding in the log: it means the
                    # budget ran out before the agent was ready to write.
                    record = self.log.info if not requests else self.log.warn
                    record("lookup.done", agent=self.agent.name,
                           calls=len(self.toolbox.calls),
                           rounds=self.toolbox.used_rounds,
                           answered=not requests)
                return payload
            self.toolbox.round(requests)
            self._refill()

    def _refill(self) -> None:
        self.protocol.fill(self.toolbox.protocol())
        notice = self.SPENT if self.toolbox.spent else ""
        room = max(0, self.results.room() - len(self.HEADER) - len(notice))
        self.results.fill(self.HEADER + self.toolbox.transcript(room) + notice)


def requested(payload: Any) -> list[Any]:
    """The lookups a model asked for, in whatever shape it asked.

    A payload carrying both an answer and a lookup is read as a lookup: an
    agent that still wanted to check something was not finished, and the
    answer it guessed at is the one this exists to prevent.
    """
    if not isinstance(payload, dict):
        return []
    for key in REQUEST_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list) and value:
            return list(value)
    if str(payload.get("tool") or "") in TOOLS:
        return [payload]
    return []


def _path(request: Any, default: str = "") -> str:
    if not isinstance(request, dict):
        return default
    raw = request.get("path") or request.get("file") or request.get("directory")
    return str(raw).strip().lstrip("/") if raw else default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "?"
    return f"{size}b" if size < 10_000 else f"{size // 1000}kb"
