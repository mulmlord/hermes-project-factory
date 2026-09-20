"""Work journal: what each agent did, in a form the next agent can read.

Agents are stateless, so without this a ticket has no way to know what earlier
tickets built, what they named things, or which approaches were already tried
and abandoned. Every ticket outcome appends one entry here. The next ticket's
prompt gets back the entries that touch its dependencies and its files, plus an
index naming the current owner of every path.

The prose summary is model-written and can be wrong or stale; the paths, gates
and commands are extracted from what actually ran, so the rendering leans on
those. The journal is an index telling an agent *which files to read*, never a
substitute for reading them.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from .log import Logger, NULL

#: Share of a prompt's journal budget that only failed or abandoned work may
#: spend. Small enough that history of what was built still dominates, large
#: enough that a dead end always reaches the agent about to walk into it.
FAILURE_RESERVE = 0.3


@dataclass
class Entry:
    ticket: str
    title: str
    kind: str = "implementation"
    outcome: str = "completed"      # completed | failed | abandoned
    summary: str = ""               # one line of model prose, never trusted alone
    files: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)   # commands that really passed
    gates: list[str] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)
    diagnosis: str = ""             # why it failed: the half worth most to the next agent
    commit: str = ""
    at: float = field(default_factory=time.time)

    def render(self) -> str:
        head = f"{self.ticket} [{self.outcome}] {self.title}"
        lines = [head + (f" ({self.commit})" if self.commit else "")]
        if self.summary:
            lines.append(f"  did: {self.summary}")
        if self.files:
            lines.append(f"  wrote: {', '.join(self.files[:8])}")
        if self.deleted:
            lines.append(f"  deleted: {', '.join(self.deleted[:5])}")
        if self.verified:
            lines.append(f"  proven by: {'; '.join(self.verified[:3])}")
        if self.diagnosis:
            lines.append(f"  failed because: {self.diagnosis}")
        if self.advisories:
            lines.append(f"  review noted: {'; '.join(self.advisories[:2])}")
        return "\n".join(lines)


class Journal:
    def __init__(self, root: Path, log: Logger = NULL):
        self.root = Path(root)
        self.path = self.root / ".hermes" / "state" / "journal.jsonl"
        self.log = log
        self.entries: list[Entry] = list(_read(self.path))

    # -- writing ---------------------------------------------------------
    def append(self, entry: Entry) -> Entry:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(entry), default=str) + "\n")
        self.entries.append(entry)
        self.mirror()
        self.log.info("journal.record", ticket=entry.ticket, outcome=entry.outcome,
                      files=len(entry.files))
        return entry

    def record(self, ticket, outcome: dict, *, status: str | None = None,
               commit: str = "") -> Entry:
        """Turn one ``execute_ticket`` verdict into an entry."""
        change = outcome.get("change")
        verification = outcome.get("verification")
        review = outcome.get("review") or {}
        return self.append(Entry(
            ticket=ticket.id,
            title=ticket.title,
            kind=ticket.kind,
            outcome=status or outcome.get("status", "failed"),
            summary=str(change.summary if change else "")[:240],
            files=[_norm(p) for p in (change.written if change else [])][:12],
            deleted=[_norm(p) for p in (change.deleted if change else [])][:6],
            verified=[r.command for r in getattr(verification, "results", []) if r.ok][:4],
            gates=[_gate(g) for g in getattr(verification, "gates", [])],
            advisories=[str(a)[:160] for a in (review.get("advisory") or [])][:3],
            diagnosis=str((outcome.get("diagnosis") or {}).get("diagnosis", ""))[:300],
            commit=commit,
        ))

    def mirror(self, limit: int = 200) -> Path:
        """Human-readable copy, so the workspace reads without Hermes internals."""
        path = self.root / "docs" / "generated" / "worklog.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n\n".join(
            f"### {time.strftime('%Y-%m-%d %H:%M', time.localtime(e.at))} — "
            f"{e.ticket} {e.title}\n\n```\n{e.render()}\n```"
            for e in self.entries[-limit:])
        path.write_text("# Work log\n\nWhat each agent did, oldest first. "
                        "Generated by Hermes.\n\n" + body + "\n")
        return path

    # -- reading ---------------------------------------------------------
    def latest(self) -> list[Entry]:
        """One entry per ticket, newest wins: a superseded attempt is not what stands."""
        newest: dict[str, Entry] = {}
        for entry in self.entries:
            newest[entry.ticket] = entry
        return list(newest.values())

    def owners(self) -> dict[str, Entry]:
        """Newest successful writer of each path. The one claim that cannot go stale."""
        index: dict[str, Entry] = {}
        for entry in self.entries:
            if entry.outcome == "completed":
                for path in entry.files:
                    index[path] = entry
        return index

    def relevant(self, ticket, board, budget: int = 4000) -> list[Entry]:
        """The entries *this* ticket needs, most relevant first.

        Selecting by recency alone would re-create the context overflow this is
        meant to solve, so the dependency graph and file overlap choose first
        and recency only spends what budget is left. The order returned is
        relevance, not time: callers that want a chronology sort it themselves.
        """
        wanted = _dependency_closure(board, ticket)
        artifacts = {_norm(a) for a in ticket.artifacts}

        def rank(entry: Entry) -> int:
            if entry.ticket in (ticket.id, ticket.parent):
                return 0                                # this ticket's own history
            if entry.ticket in wanted:
                return 1                                # what this ticket builds on
            if artifacts.intersection(entry.files):
                return 2                                # last agent to touch these files
            if entry.outcome != "completed":
                return 3                                # a dead end worth not repeating
            return 4

        ranked = sorted(self.latest(), key=lambda e: (rank(e), -e.at))
        chosen: dict[str, Entry] = {}
        used = 0

        def spend(entries: list[Entry], limit: int) -> None:
            nonlocal used
            for entry in entries:
                if entry.ticket in chosen:
                    continue
                size = len(entry.render())
                if used + size > limit:
                    continue   # skip it, never stop: what follows may still fit
                used += size
                chosen[entry.ticket] = entry

        # Dead ends rank below three classes of success, so on a board with any
        # real dependency history they are what the budget runs out before —
        # exactly the entries this journal exists to carry forward. Hence the
        # reserve: failures get first call on part of the budget, and the rest
        # of the ranking then spends what they left.
        spend([e for e in ranked if e.outcome != "completed"],
              int(budget * FAILURE_RESERVE))
        spend(ranked, budget)
        if not chosen and ranked:
            chosen[ranked[0].ticket] = ranked[0]   # one oversized entry beats none
        return sorted(chosen.values(), key=lambda e: (rank(e), -e.at))

    def block(self, ticket, board, budget: int = 4000) -> str:
        """The prompt fragment. Empty until some work has actually happened."""
        entries = self.relevant(ticket, board, budget)
        if not entries:
            return ""
        entries.sort(key=lambda e: e.at)   # a log reads oldest first
        paths = {p for e in entries for p in e.files} | {_norm(a) for a in ticket.artifacts}
        index = [f"  {path} — last written by {owner.ticket} ({owner.title[:48]})"
                 for path, owner in self.owners().items() if path in paths][:30]
        return (
            "WORK ALREADY DONE BY OTHER AGENTS (build on it; do not redo it, "
            "rename its interfaces, or repeat an approach recorded as failed):\n"
            + "\n".join(e.render() for e in entries)
            + ("\n\nFILES THAT ALREADY EXIST (read the current contents given below "
               "before you replace them):\n" + "\n".join(index) if index else "")
        )

    def suggested_files(self, ticket, board, limit: int = 6) -> list[str]:
        """Paths prior work says matter here, most relevant first.

        This is what stops the implementer guessing its context from ticket
        artifacts that may not exist yet, so the order has to follow relevance:
        a dependency's file beats whatever an unrelated ticket happened to
        write last.
        """
        paths: list[str] = []
        for entry in self.relevant(ticket, board):
            for path in entry.files:
                if path not in paths:
                    paths.append(path)
        return paths[:limit]


def _read(path: Path) -> Iterator[Entry]:
    if not path.is_file():
        return
    known = set(Entry.__dataclass_fields__)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:                      # a torn line from a crash must not kill the run
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict) and raw.get("ticket"):
            yield Entry(**{k: v for k, v in raw.items() if k in known})


def _dependency_closure(board, ticket) -> set[str]:
    seen: set[str] = set()
    stack = list(ticket.dependencies)
    while stack:
        current = stack.pop()
        if current in seen or current not in board.tickets:
            continue
        seen.add(current)
        stack.extend(board.tickets[current].dependencies)
    return seen


def _gate(gate) -> str:
    return f"{gate.name}:" + ("skip" if gate.skipped else "pass" if gate.passed else "FAIL")


def _norm(path: str) -> str:
    return str(path).replace("\\", "/").lstrip("/")
