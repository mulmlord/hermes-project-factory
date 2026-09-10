"""Kanban ticket engine.

The JSON store under ``.hermes/state/tickets.json`` is the source of truth;
markdown files under ``tickets/<status>/`` are rendered for humans on every
write so the workspace stays readable without Hermes internals.
"""
from __future__ import annotations

import json
import subprocess
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

STATUSES = ["backlog", "ready", "active", "review", "blocked", "completed", "abandoned"]
#: Only these directories are materialised (spec section 4).
BOARD_DIRS = ["backlog", "active", "blocked", "review", "completed"]

KINDS = ("research", "experiment", "implementation", "test", "optimization",
         "documentation", "refactor")


@dataclass
class Ticket:
    id: str
    title: str
    objective: str = ""
    context: str = ""
    kind: str = "implementation"
    status: str = "backlog"
    dependencies: list[str] = field(default_factory=list)
    acceptance: list[str] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    priority: int = 3           # 1 highest .. 5 lowest
    risk: int = 3               # 1 low .. 5 high
    cost: int = 3               # 1 cheap .. 5 expensive
    info_gain: int = 3          # 1 low .. 5 high
    depth: int = 0              # decomposition depth
    parent: str | None = None
    attempts: int = 0
    notes: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def render(self) -> str:
        def block(name: str, items: Iterable[str]) -> str:
            lines = "\n".join(f"- {i}" for i in items) or "- (none)"
            return f"## {name}\n{lines}\n"

        return (
            f"# {self.id}: {self.title}\n\n"
            f"- **Status**: {self.status}\n"
            f"- **Kind**: {self.kind}\n"
            f"- **Priority**: {self.priority}  **Risk**: {self.risk}  "
            f"**Cost**: {self.cost}  **Info gain**: {self.info_gain}\n"
            f"- **Attempts**: {self.attempts}\n\n"
            f"## Objective\n{self.objective or self.title}\n\n"
            f"## Context\n{self.context or '(none)'}\n\n"
            + block("Dependencies", self.dependencies)
            + block("Acceptance Criteria", self.acceptance)
            + block("Validation Commands", self.validation)
            + block("Expected Artifacts", self.artifacts)
            + block("Notes", self.notes)
        )


def split_validation(entries: list[str]) -> tuple[list[str], list[str]]:
    """Separate runnable commands from prose the planner wrote by mistake.

    A validation entry is executed as a shell command. Planners regularly put
    an English acceptance criterion there instead, which can only ever fail,
    so it is demoted to acceptance rather than run. ``sh -n`` parses without
    executing, so nothing here runs model-written text.
    """
    commands, prose = [], []
    for entry in entries:
        checked = subprocess.run(["/bin/sh", "-n", "-c", entry], capture_output=True)
        (commands if checked.returncode == 0 else prose).append(entry)
    return commands, prose


class Board:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / ".hermes" / "state" / "tickets.json"
        self.tickets: dict[str, Ticket] = {}
        self._counter = 0
        self.load()

    # -- persistence -----------------------------------------------------
    def load(self) -> None:
        if not self.path.is_file():
            return
        raw = json.loads(self.path.read_text())
        known = set(Ticket.__dataclass_fields__)
        self.tickets = {
            t["id"]: Ticket(**{k: v for k, v in t.items() if k in known})
            for t in raw.get("tickets", [])
        }
        self._counter = raw.get("counter", len(self.tickets))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"counter": self._counter,
                   "tickets": [asdict(t) for t in self.tickets.values()]}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)
        self.render()

    def render(self) -> None:
        board = self.root / "tickets"
        for status in BOARD_DIRS:
            directory = board / status
            directory.mkdir(parents=True, exist_ok=True)
            for stale in directory.glob("*.md"):
                stale.unlink()
        for ticket in self.tickets.values():
            status = ticket.status if ticket.status in BOARD_DIRS else "backlog"
            if ticket.status == "ready":
                status = "backlog"
            elif ticket.status == "abandoned":
                status = "blocked"
            slug = re.sub(r"[^a-z0-9]+", "-", ticket.title.lower()).strip("-")[:48]
            (board / status / f"{ticket.id}-{slug or 'ticket'}.md").write_text(ticket.render())

    # -- mutation --------------------------------------------------------
    def next_id(self) -> str:
        self._counter += 1
        return f"T{self._counter:03d}"

    def add(self, **fields) -> Ticket:
        fields.setdefault("id", self.next_id())
        fields.setdefault("title", "untitled")
        fields["kind"] = fields.get("kind") if fields.get("kind") in KINDS else "implementation"
        known = set(Ticket.__dataclass_fields__)
        ticket = Ticket(**{k: v for k, v in fields.items() if k in known})
        self.tickets[ticket.id] = ticket
        return ticket

    def get(self, ticket_id: str) -> Ticket | None:
        return self.tickets.get(ticket_id)

    def move(self, ticket_id: str, status: str, note: str = "") -> Ticket:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        ticket = self.tickets[ticket_id]
        ticket.status = status
        ticket.updated = time.time()
        if note:
            ticket.notes.append(f"[{time.strftime('%H:%M:%S')}] {note}")
        return ticket

    # -- queries ---------------------------------------------------------
    def by_status(self, *statuses: str) -> list[Ticket]:
        return [t for t in self.tickets.values() if t.status in statuses]

    def counts(self) -> dict[str, int]:
        counts = {s: 0 for s in STATUSES}
        for ticket in self.tickets.values():
            counts[ticket.status] = counts.get(ticket.status, 0) + 1
        return counts

    def deps_met(self, ticket: Ticket) -> bool:
        return all(
            (dep not in self.tickets) or self.tickets[dep].status == "completed"
            for dep in ticket.dependencies
        )

    def unblock(self) -> list[Ticket]:
        """Promote backlog/blocked tickets whose dependencies are satisfied."""
        moved = []
        for ticket in self.tickets.values():
            if ticket.status in ("backlog", "blocked") and self.deps_met(ticket):
                ticket.status = "ready"
                moved.append(ticket)
            elif ticket.status == "ready" and not self.deps_met(ticket):
                ticket.status = "blocked"
        return moved

    def cycles(self) -> list[list[str]]:
        """Dependency cycles, so the planner can break them instead of hanging."""
        found: list[list[str]] = []
        colour: dict[str, int] = {}

        def visit(node: str, stack: list[str]) -> None:
            colour[node] = 1
            for dep in self.tickets[node].dependencies if node in self.tickets else []:
                if dep not in self.tickets:
                    continue
                if colour.get(dep, 0) == 1:
                    found.append(stack[stack.index(dep):] + [dep])
                elif colour.get(dep, 0) == 0:
                    visit(dep, stack + [dep])
            colour[node] = 2

        for node in list(self.tickets):
            if colour.get(node, 0) == 0:
                visit(node, [node])
        return found

    def critical_path_depth(self, ticket: Ticket, _seen: set[str] | None = None) -> int:
        """How many tickets transitively depend on this one."""
        seen = _seen if _seen is not None else set()
        if ticket.id in seen:
            return 0
        seen.add(ticket.id)
        dependents = [t for t in self.tickets.values() if ticket.id in t.dependencies]
        return len(dependents) + sum(
            self.critical_path_depth(d, seen) for d in dependents
        )

    def score(self, ticket: Ticket) -> float:
        """Higher is more urgent. Drives autonomous ticket selection."""
        return (
            (6 - ticket.priority) * 3.0            # priority dominates
            + self.critical_path_depth(ticket) * 2.0   # unblock the most work
            + ticket.info_gain * 1.0               # prefer learning early
            + (6 - ticket.cost) * 0.5              # cheap wins first
            - ticket.risk * 0.5                    # defer risky work slightly
            - ticket.attempts * 2.0                # back off repeated failures
        )

    def next_ticket(self) -> Ticket | None:
        self.unblock()
        candidates = self.by_status("ready")
        if not candidates:
            return None
        return max(candidates, key=lambda t: (self.score(t), -t.created))

    def progress(self) -> float:
        if not self.tickets:
            return 0.0
        done = len(self.by_status("completed", "abandoned"))
        return done / len(self.tickets)

    def summary(self, limit: int = 25) -> str:
        rows = sorted(self.tickets.values(), key=lambda t: t.id)[:limit]
        lines = [f"{t.id:<6} {t.status:<10} {t.kind:<14} {t.title[:52]}" for t in rows]
        counts = ", ".join(f"{k}={v}" for k, v in self.counts().items() if v)
        return "\n".join(lines + [f"\n{counts}  progress={self.progress():.0%}"])
