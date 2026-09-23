"""What the factory learned about its own retries, kept across projects.

The escalation ladder in :mod:`hermes.implement` is a hypothesis: attempt two
races more samples and reads more code, attempt three asks a reasoning role
for a whole file rather than a patch.
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

It also keeps a second, smaller record: guesses a cheap classifier made in
*shadow* about a decision the factory already makes another way, kept beside
the answer that was actually used. Nothing reads those rows back into a
decision. They are how "the classifier could triage failures" would stop
being an opinion too; see :mod:`hermes.classify`.

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

SCHEMA_VERSION = 3

#: A rung as the ledger stores it: samples raced, existing sources shown, the
#: role asked, and whether the answer was demanded as complete files rather
#: than as patches. Mirrors :class:`hermes.implement.Rung`.
Spec = tuple[int, int, str, bool]

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

#: The shadow question about a failed attempt: which of the triager's classes
#: it belongs to. The key the rows are stored under, not the text asked.
TRIAGE = "triage"


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
    whole: bool = False
    fixed: int = 0
    tried: int = 0
    seconds: float = 0.0

    @property
    def rung(self) -> Spec:
        return (self.candidates, self.files, self.role, self.whole)

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
        role += ", whole-file" if self.whole else ""
        return (f"{self.candidates}x{self.files} {role}: "
                f"{self.fixed}/{self.tried} fixed ({self.rate:.0%}, "
                f"confidence {self.confidence:.0%})")


@dataclass(frozen=True)
class Agreement:
    """How often one shadow guess matched the answer the factory used."""

    guess: str
    agreed: int = 0
    seen: int = 0

    @property
    def rate(self) -> float:
        return self.agreed / self.seen if self.seen else 0.0

    @property
    def confidence(self) -> float:
        """Lower bound of the agreement rate, for the same reason as
        :attr:`Evidence.confidence`: a handful of lucky matches is not a
        classifier anyone should hand a decision to."""
        return wilson(self.agreed, self.seen)

    def describe(self) -> str:
        return (f"{self.guess}: {self.agreed}/{self.seen} agreed "
                f"({self.rate:.0%}, confidence {self.confidence:.0%})")


@dataclass(frozen=True)
class Advice:
    """A rung to try next, and why it was chosen."""

    rung: Spec
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
        self._migrate(conn)
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
                whole      INTEGER NOT NULL DEFAULT 0,
                outcome    TEXT    NOT NULL DEFAULT '',
                next       TEXT    NOT NULL DEFAULT '',
                seconds    REAL    NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS attempt_lookup
                ON attempt(signature, stack, candidates, files, role, whole);
            CREATE TABLE IF NOT EXISTS shadow (
                id         INTEGER PRIMARY KEY,
                at         REAL    NOT NULL,
                question   TEXT    NOT NULL,
                project    TEXT    NOT NULL DEFAULT '',
                stack      TEXT    NOT NULL DEFAULT '',
                kind       TEXT    NOT NULL DEFAULT '',
                signature  TEXT    NOT NULL DEFAULT '',
                attempt    INTEGER NOT NULL DEFAULT 1,
                guess      TEXT    NOT NULL,
                confidence REAL    NOT NULL DEFAULT 0,
                answer     TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS shadow_lookup ON shadow(question, guess);
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
        """)
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('schema', ?)",
                     (str(SCHEMA_VERSION),))

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Carry a ledger written by an older factory onto this schema.

        This is the one file that outlives every workspace, so a rung that
        gains a dimension has to reach the rows already in it: starting the
        record again would throw away exactly the history the ladder is
        judged against. A row written before the format rung existed was
        asked in whatever format its run configured, which is what 0 means --
        the same thing it means for a ladder that never switches format.
        """
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(attempt)")}
        if columns and "whole" not in columns:
            conn.execute("ALTER TABLE attempt ADD COLUMN whole"
                         " INTEGER NOT NULL DEFAULT 0")
            # The lookup index is keyed on the rung, which just grew a column;
            # dropping it here is what lets the schema below rebuild it.
            conn.execute("DROP INDEX IF EXISTS attempt_lookup")
            self.log.info("ledger.migrated", path=str(self.path),
                          schema=SCHEMA_VERSION)

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
    def record(self, signature: str, rung: Spec, outcome: str, *,
               project: str = "", stack: str = "", kind: str = "",
               attempt: int = 1, following: str = "", seconds: float = 0.0) -> None:
        """Record that *rung*, asked after *signature*, ended in *outcome*.

        *signature* is the failure this attempt was handed and is empty for a
        first try -- which is worth keeping even though nothing escalates on
        it, because it is the base rate every escalation is measured against.
        """
        if not self.ok:
            return
        candidates, files, role, whole = rung
        try:
            self._conn().execute(
                "INSERT INTO attempt(at, project, stack, kind, signature, attempt,"
                " candidates, files, role, whole, outcome, next, seconds)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), project[:80], stack[:40], kind[:40], signature[:80],
                 attempt, candidates, files, role[:40], int(whole), outcome,
                 following[:80], round(seconds, 2)))
        except sqlite3.Error as exc:
            self.log.warn("ledger.write_failed", error=str(exc)[:200])

    def record_guess(self, question: str, guess: str, confidence: float,
                     answer: str, *, project: str = "", stack: str = "",
                     kind: str = "", signature: str = "", attempt: int = 1) -> None:
        """Record a shadow *guess* at *question* beside the *answer* used."""
        if not self.ok:
            return
        try:
            self._conn().execute(
                "INSERT INTO shadow(at, question, project, stack, kind, signature,"
                " attempt, guess, confidence, answer) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (time.time(), question[:40], project[:80], stack[:40], kind[:40],
                 signature[:80], attempt, guess[:40], round(confidence, 4),
                 answer[:40]))
        except sqlite3.Error as exc:
            self.log.warn("ledger.write_failed", error=str(exc)[:200])

    # -- reading ---------------------------------------------------------
    def agreement(self, question: str, *, above: float = 0.0) -> list[Agreement]:
        """How often each shadow guess at *question* matched, most-seen first.

        Grouped by what was guessed rather than by what was true, because
        the question a promotion has to answer is "when it says X, is it
        right" -- and restricted to guesses at or *above* a confidence, since
        a classifier that is only ever handed its confident answers is judged
        on those alone.
        """
        if not self.ok:
            return []
        try:
            rows = self._conn().execute(
                "SELECT guess, COUNT(*) AS seen, SUM(guess = answer) AS agreed"
                " FROM shadow WHERE question = ? AND confidence >= ?"
                " GROUP BY guess ORDER BY seen DESC, guess",
                (question, above)).fetchall()
            return [Agreement(row["guess"], row["agreed"] or 0, row["seen"])
                    for row in rows]
        except sqlite3.Error as exc:
            self.log.warn("ledger.read_failed", error=str(exc)[:200])
            return []

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
            f"SELECT candidates, files, role, whole, COUNT(*) AS tried,"
            f" SUM(outcome = ?) AS fixed, AVG(seconds) AS seconds"
            f" FROM attempt WHERE {where}"
            f" GROUP BY candidates, files, role, whole", [FIXED] + params).fetchall()
        return [Evidence(candidates=row["candidates"], files=row["files"],
                         role=row["role"], whole=bool(row["whole"]),
                         fixed=row["fixed"] or 0, tried=row["tried"],
                         seconds=row["seconds"] or 0.0)
                for row in rows]

    def advice(self, signature: str, default: Spec,
               options: list[Spec], *, stack: str = "") -> Advice:
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
            dropped += conn.execute(
                "DELETE FROM shadow WHERE id NOT IN"
                " (SELECT id FROM shadow ORDER BY id DESC LIMIT ?)",
                (keep,)).rowcount or 0
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

    def record_guess(self, *_args, **_kwargs) -> None:
        return None

    def evidence(self, *_args, **_kwargs) -> list[Evidence]:
        return []

    def agreement(self, *_args, **_kwargs) -> list[Agreement]:
        return []

    def advice(self, signature: str, default: Spec,
               options: list[Spec], **_kwargs) -> Advice:
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
