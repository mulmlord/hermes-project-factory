"""What the factory learned about its own retries, kept across projects.

The escalation ladder in :mod:`hermes.implement` is a hypothesis: attempt two
races more samples and reads more code, attempt three asks a reasoning role.
It is a reasonable hypothesis and nothing in the factory has ever checked it.
Each project starts from the same fixed guess, and everything it discovers
about which second try actually rescues which kind of failure dies with the
workspace.

This is the store that outlives the workspace. One row per attempt: the
failure it was handed, how it was asked, and whether that worked. With enough
rows, "a build failure at attempt two is better served by the reasoning role
than by two more samples" stops being an opinion and becomes a query.

The policy on top of it is deliberately timid, because a learned ladder that
is wrong is worse than a fixed one that is merely unproven:

* a rung is only considered once it has been watched :data:`MIN_EVIDENCE`
  times under that signature;
* the ladder's own choice is only overruled when *it* has been watched that
  often too, so the ledger never argues from ignorance of the thing it is
  replacing;
* the comparison is on the lower bound of the pass rate, not the rate, so two
  lucky successes do not outrank forty observed ones;
* where two rungs are within :data:`MARGIN` of each other the cheaper one
  wins, which is how a rung that escalates for no measured benefit stops
  costing three model calls.

Cold, with no rows at all, every one of those rules falls through to the
ladder, and the factory behaves exactly as it did before this existed.

The database lives outside every workspace -- it is the factory's memory, not
any one project's -- and it is advisory: a lost or corrupt ledger costs the
learning and nothing else, which is why every method here swallows its own
storage errors rather than letting them reach a ticket.
"""
from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .log import Logger, NULL

SCHEMA_VERSION = 1

#: Times a rung must have been watched under a signature before its record is
#: allowed to mean anything. Low enough to learn inside a few projects, high
#: enough that one unlucky ticket does not rewrite the ladder.
MIN_EVIDENCE = 8

#: Pass-rate difference below which two rungs are treated as equally good, so
#: that the cheaper of them wins. Five points is well inside the noise of the
#: sample sizes this works with.
MARGIN = 0.05

#: Rows kept. An attempt is a few hundred bytes and a project is a few hundred
#: attempts, so this is years of use and a file measured in megabytes.
KEEP = 50_000

#: Outcomes. *fixed* means the attempt cleared the gates it was handed,
#: *failed* that it produced a new failure, *abandoned* that the run stopped
#: asking -- triage sent the ticket to be decomposed, researched or rolled
#: back, which says nothing about the rung and is never counted for or
#: against it.
FIXED, FAILED, ABANDONED = "fixed", "failed", "abandoned"


def default_path() -> Path:
    """Where the ledger lives when the configuration does not say.

    ``HERMES_LEDGER`` first, so a test or a second factory on one machine can
    keep its own; then the XDG state directory, which is where a file that is
    neither configuration nor a cache belongs.
    """
    if override := os.environ.get("HERMES_LEDGER"):
        return Path(override).expanduser()
    state = os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state")
    return Path(state) / "hermes" / "factory.db"


@dataclass(frozen=True)
class Evidence:
    """What the ledger has seen of one rung under one signature."""

    candidates: int = 1
    files: int = 6
    role: str = ""
    fixed: int = 0
    tried: int = 0
    seconds: float = 0.0

    @property
    def rung(self) -> tuple[int, int, str]:
        return (self.candidates, self.files, self.role)

    @property
    def rate(self) -> float:
        return self.fixed / self.tried if self.tried else 0.0

    @property
    def confidence(self) -> float:
        """The lower bound of the pass rate, which is what gets compared.

        Wilson rather than the plain rate: at these sample sizes the plain
        rate says 1.0 after two successes, and a ladder that reorders itself
        on two successes is a ladder that reorders itself on noise.
        """
        return wilson(self.fixed, self.tried)

    @property
    def cost(self) -> int:
        """Model calls one attempt at this rung spends, which breaks ties."""
        return self.candidates

    def describe(self) -> str:
        role = self.role or "default role"
        return (f"{self.candidates}x{self.files} {role}: "
                f"{self.fixed}/{self.tried} fixed ({self.rate:.0%}, "
                f"confidence {self.confidence:.0%})")


@dataclass(frozen=True)
class Advice:
    """A rung to try next, and why it was chosen."""

    rung: tuple[int, int, str]
    reason: str = "ladder"
    confidence: float = 0.0
    tried: int = 0

    @property
    def learned(self) -> bool:
        return self.reason != "ladder"


def wilson(passes: int, total: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for *passes* of *total*."""
    if total <= 0:
        return 0.0
    rate = passes / total
    divisor = 1 + z * z / total
    centre = rate + z * z / (2 * total)
    spread = z * math.sqrt((rate * (1 - rate) + z * z / (4 * total)) / total)
    return max(0.0, (centre - spread) / divisor)


class Ledger:
    """The factory's record of which retry rescued which failure.

    Safe to share: one connection per thread, WAL, and a busy timeout, since
    two projects may be building on one machine and both will write here.
    """

    def __init__(self, path: Path, log: Logger = NULL):
        self.path = Path(path)
        self.log = log
        self._local = threading.local()
        self.ok = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._setup()
        except (sqlite3.Error, OSError) as exc:
            # A factory that cannot open its ledger still builds projects; it
            # just stops learning from them.
            self.ok = False
            self.log.warn("ledger.unavailable", path=str(self.path),
                          error=str(exc)[:200])

    # -- plumbing --------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    def _setup(self) -> None:
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS attempt (
                id         INTEGER PRIMARY KEY,
                at         REAL    NOT NULL,
                project    TEXT    NOT NULL DEFAULT '',
                stack      TEXT    NOT NULL DEFAULT '',
                kind       TEXT    NOT NULL DEFAULT '',
                signature  TEXT    NOT NULL DEFAULT '',
                attempt    INTEGER NOT NULL DEFAULT 1,
                candidates INTEGER NOT NULL DEFAULT 1,
                files      INTEGER NOT NULL DEFAULT 6,
                role       TEXT    NOT NULL DEFAULT '',
                outcome    TEXT    NOT NULL DEFAULT '',
                next       TEXT    NOT NULL DEFAULT '',
                seconds    REAL    NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS attempt_lookup
                ON attempt(signature, stack, candidates, files, role);
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
        """)
        conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('schema', ?)",
                     (str(SCHEMA_VERSION),))

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # -- writing ---------------------------------------------------------
    def record(self, signature: str, rung: tuple[int, int, str], outcome: str, *,
               project: str = "", stack: str = "", kind: str = "",
               attempt: int = 1, following: str = "", seconds: float = 0.0) -> None:
        """Record that *rung*, asked after *signature*, ended in *outcome*.

        *signature* is the failure this attempt was handed and is empty for a
        first try -- which is worth keeping even though nothing escalates on
        it, because it is the base rate every escalation is measured against.
        """
        if not self.ok:
            return
        candidates, files, role = rung
        try:
            self._conn().execute(
                "INSERT INTO attempt(at, project, stack, kind, signature, attempt,"
                " candidates, files, role, outcome, next, seconds)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), project[:80], stack[:40], kind[:40], signature[:80],
                 attempt, candidates, files, role[:40], outcome, following[:80],
                 round(seconds, 2)))
        except sqlite3.Error as exc:
            self.log.warn("ledger.write_failed", error=str(exc)[:200])

    # -- reading ---------------------------------------------------------
    def evidence(self, signature: str, *, stack: str = "") -> list[Evidence]:
        """Every rung ever tried against *signature*, most-watched first.

        Restricted to *stack* when that narrower question has been asked
        enough times to answer, because a repair is shaped by its toolchain --
        a failing C++ link and a failing pytest import are not the same
        problem wearing different words. Below that threshold the narrower
        slice would be a handful of rows, so the broader one is the better
        evidence even though it is the vaguer question.
        """
        if not self.ok:
            return []
        try:
            rows = self._rungs(signature, stack) if stack else []
            if sum(row.tried for row in rows) < MIN_EVIDENCE:
                rows = self._rungs(signature, "")
            return sorted(rows, key=lambda row: -row.tried)
        except sqlite3.Error as exc:
            self.log.warn("ledger.read_failed", error=str(exc)[:200])
            return []

    def _rungs(self, signature: str, stack: str) -> list[Evidence]:
        where = "signature = ? AND outcome != ?"
        params: list = [signature, ABANDONED]
        if stack:
            where += " AND stack = ?"
            params.append(stack)
        rows = self._conn().execute(
            f"SELECT candidates, files, role, COUNT(*) AS tried,"
            f" SUM(outcome = ?) AS fixed, AVG(seconds) AS seconds"
            f" FROM attempt WHERE {where}"
            f" GROUP BY candidates, files, role", [FIXED] + params).fetchall()
        return [Evidence(candidates=row["candidates"], files=row["files"],
                         role=row["role"], fixed=row["fixed"] or 0,
                         tried=row["tried"], seconds=row["seconds"] or 0.0)
                for row in rows]

    def advice(self, signature: str, default: tuple[int, int, str],
               options: list[tuple[int, int, str]], *, stack: str = "") -> Advice:
        """Which rung to try against *signature*, out of *options*.

        Returns *default* -- the ladder's own answer -- unless the evidence
        clears every bar in this module's docstring. The caller is told which
        it got, because a ladder that quietly rearranges itself is a factory
        nobody can debug.
        """
        seen = {row.rung: row for row in self.evidence(signature, stack=stack)}
        standard = seen.get(default)
        if standard is None or standard.tried < MIN_EVIDENCE:
            # Nothing is known about the rung that would otherwise run, so
            # there is nothing to overrule. Running it is also how it becomes
            # known.
            return Advice(default)

        ranked = [seen[rung] for rung in options
                  if rung in seen and seen[rung].tried >= MIN_EVIDENCE]
        if not ranked:
            return Advice(default)
        # Best lower bound wins; inside the margin, fewest model calls wins.
        best = max(ranked, key=lambda row: (round(row.confidence / MARGIN),
                                            -row.cost, row.confidence))
        if best.rung == default:
            return Advice(default)
        cheaper = best.cost < standard.cost
        if best.confidence < standard.confidence + MARGIN and not cheaper:
            return Advice(default)
        reason = "cheaper for the same result" if cheaper else "better record"
        return Advice(best.rung, reason, best.confidence, best.tried)

    def signatures(self, limit: int = 20) -> list[tuple[str, int, float]]:
        """The failures seen most often, with how often they were repaired."""
        if not self.ok:
            return []
        try:
            rows = self._conn().execute(
                "SELECT signature, COUNT(*) AS tried, AVG(outcome = ?) AS rate"
                " FROM attempt WHERE signature != '' AND outcome != ?"
                " GROUP BY signature ORDER BY tried DESC LIMIT ?",
                (FIXED, ABANDONED, limit)).fetchall()
            return [(row["signature"], row["tried"], row["rate"] or 0.0)
                    for row in rows]
        except sqlite3.Error:
            return []

    def stats(self) -> dict[str, object]:
        if not self.ok:
            return {"available": False}
        try:
            row = self._conn().execute(
                "SELECT COUNT(*) AS rows, COUNT(DISTINCT project) AS projects,"
                " COUNT(DISTINCT signature) AS signatures,"
                " AVG(outcome = ?) AS fixed, MIN(at) AS first"
                " FROM attempt", (FIXED,)).fetchone()
            return {"available": True, "path": str(self.path),
                    "attempts": row["rows"], "projects": row["projects"],
                    "signatures": row["signatures"],
                    "repaired": round(row["fixed"] or 0.0, 3),
                    "since": row["first"] or 0.0}
        except sqlite3.Error:
            return {"available": False}

    def prune(self, keep: int = KEEP) -> int:
        """Drop the oldest rows beyond *keep*, newest always retained."""
        if not self.ok:
            return 0
        try:
            conn = self._conn()
            cursor = conn.execute(
                "DELETE FROM attempt WHERE id NOT IN"
                " (SELECT id FROM attempt ORDER BY id DESC LIMIT ?)", (keep,))
            dropped = cursor.rowcount or 0
            if dropped:
                conn.execute("VACUUM")
            return dropped
        except sqlite3.Error:
            return 0


class NullLedger:
    """What the factory uses when learning is switched off.

    A stand-in rather than a ``None`` check at every call site: the loop
    should not have to know whether anything is being remembered.
    """

    ok = False
    path = Path()

    def record(self, *_args, **_kwargs) -> None:
        return None

    def evidence(self, *_args, **_kwargs) -> list[Evidence]:
        return []

    def advice(self, signature: str, default: tuple[int, int, str],
               options: list[tuple[int, int, str]], **_kwargs) -> Advice:
        return Advice(default)

    def signatures(self, limit: int = 20) -> list[tuple[str, int, float]]:
        return []

    def stats(self) -> dict[str, object]:
        return {"available": False}

    def prune(self, keep: int = KEEP) -> int:
        return 0

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_) -> None:
        return None


def open_ledger(config=None, log: Logger = NULL) -> Ledger | NullLedger:
    """The ledger this configuration asks for, or a stand-in for none."""
    if config is not None and not config.get("execution.ledger", True):
        return NullLedger()
    configured = config.get("execution.ledger_path") if config is not None else ""
    path = Path(configured).expanduser() if configured else default_path()
    ledger = Ledger(path, log)
    return ledger if ledger.ok else NullLedger()
