"""The project context database: what every agent learned, kept queryable.

Agents here are stateless, so everything one knows has to reach the next one
through the workspace. Today that handover is files -- the journal, the
context document, the sources block -- and a file costs the same to hand over
whether or not it changed since the last agent read it. On a project of any
size that is most of a prompt spent re-reading text the previous agent was
already shown.

This is the store that makes the handover incremental. Two kinds of thing go
in, both stamped with a monotonically increasing revision:

* **records** -- a log line, a gate result, a decision, a remembered fact,
  the outcome of a ticket. What an agent *observed*.
* **chunks** -- files, split at their own boundaries and keyed by the hash of
  their contents. What the project *is*.

A reader remembers the revision it last saw (its *cursor*) and asks for
everything after it. Because chunks are content-keyed, a one-function change
to a 900-line file is one chunk in that answer and not the file; because
records are rows, "what happened to src/engine.py" is an index lookup and not
a scan of the log. Cold readers pass cursor 0 and get the history instead,
so nothing has to know whether it has been here before.

SQLite, and nothing but the standard library: every generated project runs
this in its build, so it has to be present everywhere, need no server,
survive a crash mid-write, and serve readers while the loop writes -- which
is WAL mode and `sqlite3`. A directory of JSON files would need a hand-built
index for every query below, and an append-only log cannot answer "what
changed inside this file" without reading all of it, which is the cost this
exists to remove.

The database is an index over durable things, never the only copy of one.
The files are on disk and the journal is still JSONL; a lost database is
rebuilt with :meth:`ContextDB.sync_tree`, which is why it is not committed.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Iterator

from .log import Logger, NULL

#: Where the database lives inside a workspace. Beside `.hermes/state` rather
#: than inside it: `state/` is copied wholesale into every checkpoint, and
#: copying an open WAL database is how you get a torn one. :meth:`backup`
#: exists for that instead.
DB_PATH = Path(".hermes") / "context.db"

SCHEMA_VERSION = 1

#: Chunking. A chunk is one thing an agent might need -- a function, a section
#: -- so the cut goes on a boundary the text itself declares, and these sizes
#: only stop a stretch with no boundary in it becoming one enormous chunk.
MAX_LINES = 60
MAX_CHARS = 2400

#: A file bigger than this is data, not source: chunking it would fill the
#: database with rows no agent will ever be shown.
MAX_FILE_BYTES = 400_000

#: Directories :meth:`sync_tree` never walks. Same list the lookup tools use,
#: and for the same reason: none of it is the project.
SKIP = frozenset({".git", ".hermes", "node_modules", ".venv", "venv", "artifacts",
                  "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
                  ".tox", ".eggs", "build", "dist", "target"})

#: Line shapes that start a new chunk even mid-paragraph. Deliberately
#: cross-language and deliberately shallow: guessing a boundary wrong costs a
#: slightly worse split, while parsing per language costs a parser per
#: language.
_BOUNDARY = re.compile(
    r"^(#{1,6}\s|@|def\s|class\s|async\s+def\s|func\s|fn\s|function\s|impl\s|"
    r"struct\s|enum\s|trait\s|interface\s|module\s|package\s|namespace\s|"
    r"public\s|private\s|protected\s|static\s|export\s|const\s|var\s|let\s|"
    r"type\s|template\s|/\*\*|---)"
)

#: Room a rendered delta sets aside for its own section headings and for the
#: line naming whatever did not fit, so that a budget which is a prompt
#: block's room stays one.
HEADINGS = 400


# -- rows --------------------------------------------------------------------


@dataclass
class Record:
    """One thing an agent observed, as it is read back."""

    id: int = 0
    rev: int = 0
    at: float = 0.0
    kind: str = "note"
    agent: str = ""
    ticket: str = ""
    title: str = ""
    body: str = ""
    key: str = ""
    weight: int = 5
    paths: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        head = f"[{self.kind}]"
        if self.ticket:
            head += f" {self.ticket}"
        if self.agent:
            head += f" ({self.agent})"
        line = f"{head} {self.title}".rstrip()
        if self.paths:
            line += f"  -- {', '.join(self.paths[:6])}"
        body = self.body.strip()
        return f"{line}\n{_indent(body)}" if body else line


@dataclass
class Chunk:
    """One piece of one file, as stored."""

    path: str = ""
    ord: int = 0
    hash: str = ""
    heading: str = ""
    first_line: int = 0
    last_line: int = 0
    body: str = ""
    rev: int = 0
    at: float = 0.0

    def render(self) -> str:
        return (f"--- {self.path} lines {self.first_line}-{self.last_line}"
                f"  [{self.hash}]\n{self.body}")


@dataclass
class Sync:
    """What ingesting one file actually changed."""

    path: str
    rev: int = 0
    added: int = 0
    removed: int = 0
    kept: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.rev)


@dataclass
class Hit:
    """A search result, from either table."""

    kind: str = ""
    ref: str = ""
    path: str = ""
    title: str = ""
    body: str = ""
    rev: int = 0

    def render(self) -> str:
        where = self.path or self.ref
        return f"{where}: {self.title}\n{_indent(self.body.strip())}".rstrip()


@dataclass
class Gone:
    """A file that was here at the reader's cursor and is not here now."""

    path: str = ""
    rev: int = 0

    def render(self) -> str:
        return f"- {self.path}"


@dataclass
class Delta:
    """Everything that happened after a reader's cursor.

    :meth:`render` is where the budget is spent, and it is also what decides
    how far the cursor may advance: anything dropped for want of room has to
    still be unseen next time, or the budget would silently become data loss.
    """

    since: int = 0
    head: int = 0
    records: list[Record] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    removed: list["Gone"] = field(default_factory=list)
    text: str = ""
    dropped: int = 0
    #: The revision the reader may honestly claim to have seen.
    through: int = 0

    @property
    def empty(self) -> bool:
        return not (self.records or self.chunks or self.removed)

    def render(self, budget: int = 4000) -> str:
        """Lay the delta out for a prompt, within *budget* characters.

        The budget is spent a whole revision at a time, and the cursor stops
        at the last revision that fitted entirely. Half a revision would be
        the worst of both: the reader could not claim it, so it would be
        offered the same half again on its next call, forever.

        One revision that does not fit on its own is the exception, because
        refusing it is exactly that forever. Then as much of it as fits is
        shown, the rest is named rather than quoted, and the cursor moves past
        it anyway -- which costs nothing that cannot be recovered, since a
        named file is still on disk and one :func:`read_file` away.
        """
        self.through, self.dropped, self.text = self.head, 0, ""
        if self.empty:
            return ""
        opening = (f"CONTEXT UPDATE (revisions {self.since + 1}-{self.head}). You "
                   f"already have everything up to revision {self.since}; only "
                   "what changed since is repeated here.")
        by_revision: dict[int, list[Any]] = {}
        for item in self.records + self.removed + self.chunks:
            by_revision.setdefault(item.rev, []).append(item)

        room = max(0, budget - len(opening) - HEADINGS)
        kept: set[int] = set()
        used = 0
        #: True when a single revision had to be broken up, which is the one
        #: case where moving the cursor leaves something behind for good.
        broken = False
        for revision in sorted(by_revision):
            group = by_revision[revision]
            cost = sum(len(item.render()) + 2 for item in group)
            if used + cost <= room:
                kept.update(id(item) for item in group)
                used += cost
                self.through = revision
                continue
            if kept:
                break  # stop on a boundary; the rest keeps for the next call
            for item in group:
                size = len(item.render()) + 2
                if used + size > room:
                    continue
                kept.add(id(item))
                used += size
            self.through, broken = revision, True
            break

        skipped = [item for item in self.records + self.removed + self.chunks
                   if id(item) not in kept]
        self.dropped = len(skipped)
        if not skipped:
            self.through = self.head

        lines = [opening]
        _section(lines, "WHAT OTHER AGENTS RECORDED:",
                 [r for r in self.records if id(r) in kept])
        _section(lines, "FILES REMOVED SINCE:",
                 [g for g in self.removed if id(g) in kept], join="\n")
        _section(lines, "CODE THAT CHANGED (only the changed parts; the rest of "
                 "each file is as you last read it):",
                 [c for c in self.chunks if id(c) in kept])
        self.text = "\n\n".join(lines)
        if skipped:
            self.text += self._notice(budget - len(self.text), skipped, broken)
        return self.text

    def _notice(self, room: int, skipped: list, broken: bool) -> str:
        """Say what was left out, in whatever room is left to say it in.

        Two situations, and telling them apart matters. What was dropped out
        of a broken revision is not coming back, because the cursor moves past
        it regardless -- so it is named, and a named file is still on disk and
        one read away. What was left behind at a revision boundary is simply
        next in line, and saying so is the whole of it. Calling the second
        kind unrecoverable would send a reader off to re-read files it is
        about to be handed anyway.

        The sentence shrinks to fit: names first, then the count alone, then
        nothing. A notice that overran the budget would defeat the budget.
        """
        lost = [item for item in skipped if broken and item.rev == self.through]
        later = self.dropped - len(lost)
        names = ", ".join(sorted(
            {getattr(item, "path", "") or getattr(item, "title", "")
             for item in lost} - {""}))
        gone = (f"{len(lost)} change(s) at revision {self.through} were too large "
                "to quote")
        named, plain = f"{gone}: {names}; read them if you need them.", f"{gone}."
        carried = f"{later} further change(s) come with your next update."

        candidates: list[str] = []
        if lost and later:
            candidates += [f"{named} {carried}", f"{plain} {carried}"]
        if lost:
            candidates += [named, plain]
        if later:
            candidates.append(carried)
        for body in candidates:
            notice = f"\n\n({body})"
            if len(notice) <= room:
                return notice
        return ""


# -- the database ------------------------------------------------------------


class ContextDB:
    """The workspace's context store. Cheap to open, safe to share.

    One connection per thread, because the factory reads this from a prefetch
    thread while the loop writes to it, and a SQLite connection is not
    something to hand between threads. WAL mode is what keeps those readers
    from blocking on the writer.
    """

    def __init__(self, path: Path, log: Logger = NULL):
        self.path = Path(path)
        self.log = log
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._open: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self.fts = True
        self._setup()

    @classmethod
    def for_workspace(cls, root: Path, log: Logger = NULL) -> "ContextDB":
        return cls(Path(root) / DB_PATH, log)

    # -- connection ------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            with self._lock:
                self._open.append(conn)
        return conn

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One write transaction. Never nest these: SQLite will not have it."""
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            for conn in self._open:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            self._open.clear()
        self._local = threading.local()

    def __enter__(self) -> "ContextDB":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def backup(self, dest: Path) -> Path:
        """A consistent copy, taken while the factory keeps writing.

        Checkpointing copies `.hermes/state` with the filesystem, which is
        exactly what must not happen to a live WAL database.
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(dest)
        try:
            self._conn().backup(target)
        finally:
            target.close()
        return dest

    # -- schema ----------------------------------------------------------
    def _setup(self) -> None:
        conn = self._conn()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);

        CREATE TABLE IF NOT EXISTS record(
            id      INTEGER PRIMARY KEY,
            rev     INTEGER NOT NULL,
            at      REAL    NOT NULL,
            kind    TEXT    NOT NULL,
            agent   TEXT    NOT NULL DEFAULT '',
            ticket  TEXT    NOT NULL DEFAULT '',
            title   TEXT    NOT NULL DEFAULT '',
            body    TEXT    NOT NULL DEFAULT '',
            key     TEXT    NOT NULL DEFAULT '',
            weight  INTEGER NOT NULL DEFAULT 5,
            paths   TEXT    NOT NULL DEFAULT '[]',
            meta    TEXT    NOT NULL DEFAULT '{}',
            stale   INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS record_rev ON record(rev);
        CREATE INDEX IF NOT EXISTS record_kind ON record(kind, rev);
        CREATE INDEX IF NOT EXISTS record_ticket ON record(ticket, rev);
        CREATE INDEX IF NOT EXISTS record_key ON record(key, stale);

        CREATE TABLE IF NOT EXISTS record_path(
            record INTEGER NOT NULL, path TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS record_path_path ON record_path(path, record);

        CREATE TABLE IF NOT EXISTS chunk(
            id         INTEGER PRIMARY KEY,
            path       TEXT    NOT NULL,
            ord        INTEGER NOT NULL,
            hash       TEXT    NOT NULL,
            heading    TEXT    NOT NULL DEFAULT '',
            first_line INTEGER NOT NULL DEFAULT 0,
            last_line  INTEGER NOT NULL DEFAULT 0,
            body       TEXT    NOT NULL DEFAULT '',
            rev        INTEGER NOT NULL,
            dead       INTEGER NOT NULL DEFAULT 0,
            at         REAL    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS chunk_live ON chunk(path, dead, ord);
        CREATE INDEX IF NOT EXISTS chunk_rev ON chunk(rev);
        CREATE INDEX IF NOT EXISTS chunk_dead ON chunk(dead);

        CREATE TABLE IF NOT EXISTS cursor(
            name    TEXT PRIMARY KEY,
            rev     INTEGER NOT NULL DEFAULT 0,
            updated REAL    NOT NULL DEFAULT 0,
            reads   INTEGER NOT NULL DEFAULT 0
        );
        """)
        conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('rev', '0')")
        conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
                     ("schema", str(SCHEMA_VERSION)))
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5("
                         "kind, ref UNINDEXED, path, title, body, "
                         "tokenize='porter unicode61')")
        except sqlite3.OperationalError:
            # Some distribution builds ship without FTS5. Search still works,
            # by scan; saying so once is better than a query that raises.
            self.fts = False
            self.log.warn("context.no_fts5", db=str(self.path))

    # -- revisions -------------------------------------------------------
    @property
    def head(self) -> int:
        """The current revision. A cursor equal to this is fully caught up."""
        row = self._conn().execute("SELECT value FROM meta WHERE key='rev'").fetchone()
        return int(row["value"]) if row else 0

    @staticmethod
    def _bump(conn: sqlite3.Connection) -> int:
        conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 "
                     "WHERE key='rev'")
        return int(conn.execute(
            "SELECT value FROM meta WHERE key='rev'").fetchone()["value"])

    # -- writing records --------------------------------------------------
    def record(self, kind: str, title: str = "", body: str = "", *,
               agent: str = "", ticket: str = "", paths: Iterable[str] = (),
               meta: dict | None = None, key: str = "",
               weight: int = 5) -> Record:
        """Store one observation and return it with its revision.

        A *key* makes the record a replacement rather than an addition: the
        previous record under that key stops being current, so an agent can
        keep a living note ("the approach we settled on for parsing") without
        every reader having to work out which of five versions is the latest.
        The superseded row stays, because how an understanding changed is
        itself worth reading.
        """
        entry = Record(kind=str(kind), title=str(title)[:300], body=str(body),
                       agent=str(agent), ticket=str(ticket), key=str(key),
                       weight=max(0, min(9, int(weight))),
                       paths=[_rel(p) for p in paths if str(p).strip()],
                       meta=dict(meta or {}), at=time.time())
        with self._write() as conn:
            entry.rev = self._bump(conn)
            if entry.key:
                conn.execute("UPDATE record SET stale = ? WHERE key = ? AND stale = 0",
                             (entry.rev, entry.key))
            cursor = conn.execute(
                "INSERT INTO record(rev, at, kind, agent, ticket, title, body, key,"
                " weight, paths, meta) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (entry.rev, entry.at, entry.kind, entry.agent, entry.ticket,
                 entry.title, entry.body, entry.key, entry.weight,
                 json.dumps(entry.paths), json.dumps(entry.meta, default=str)))
            entry.id = int(cursor.lastrowid)
            conn.executemany("INSERT INTO record_path(record, path) VALUES(?,?)",
                             [(entry.id, p) for p in entry.paths])
            self._index(conn, entry.kind, f"record:{entry.id}",
                        " ".join(entry.paths), entry.title, entry.body)
        self.log.debug("context.record", kind=entry.kind, rev=entry.rev,
                       ticket=entry.ticket or None, paths=len(entry.paths))
        return entry

    def log_event(self, event: str, *, agent: str = "", ticket: str = "",
                  **fields: Any) -> Record:
        """An agent's log line, in the database rather than only in a file.

        The JSONL log is written for an operator reading a run afterwards.
        This is the same event written for the next agent, which needs to find
        it by ticket or by path and cannot afford to read the file to do so.
        """
        detail = {k: v for k, v in fields.items() if v is not None}
        paths = detail.pop("paths", None) or ([detail["path"]] if detail.get("path") else [])
        return self.record("log", event, _fields(detail), agent=agent,
                           ticket=ticket, paths=paths, meta=detail, weight=3)

    def remember(self, key: str, body: str, *, title: str = "", agent: str = "",
                 ticket: str = "", paths: Iterable[str] = (), kind: str = "fact",
                 weight: int = 7) -> Record:
        """Write down something that stays true until it is rewritten."""
        return self.record(kind, title or key, body, agent=agent, ticket=ticket,
                           paths=paths, key=key, weight=weight)

    def recall(self, key: str) -> Record | None:
        """The current record under *key*, ignoring the ones it replaced."""
        row = self._conn().execute(
            "SELECT * FROM record WHERE key = ? AND stale = 0 "
            "ORDER BY rev DESC LIMIT 1", (str(key),)).fetchone()
        return _record(row) if row else None

    def facts(self, limit: int = 50) -> list[Record]:
        """Every living keyed record, newest first."""
        return [_record(r) for r in self._conn().execute(
            "SELECT * FROM record WHERE key <> '' AND stale = 0 "
            "ORDER BY weight DESC, rev DESC LIMIT ?", (int(limit),))]

    def records(self, *, kind: str = "", ticket: str = "", path: str = "",
                since: int = 0, limit: int = 50) -> list[Record]:
        """Records matching every filter given, newest first."""
        sql = ["SELECT r.* FROM record r"]
        args: list[Any] = []
        if path:
            sql.append("JOIN record_path p ON p.record = r.id AND p.path = ?")
            args.append(_rel(path))
        sql.append("WHERE r.rev > ?")
        args.append(int(since))
        if kind:
            sql.append("AND r.kind = ?")
            args.append(kind)
        if ticket:
            sql.append("AND r.ticket = ?")
            args.append(ticket)
        sql.append("ORDER BY r.rev DESC LIMIT ?")
        args.append(int(limit))
        return [_record(r) for r in self._conn().execute(" ".join(sql), args)]

    # -- writing files ----------------------------------------------------
    def sync_file(self, path: str, text: str) -> Sync:
        """Bring one file's chunks up to date, and report what moved.

        Chunks are matched by content, not by position, so inserting a
        function at the top of a file does not mark every chunk below it as
        changed -- that would put the whole file in the next delta and undo
        the point of storing it in pieces. A file whose contents are the same
        costs nothing at all and does not move the revision: a no-op sync
        that advanced the head would invalidate every reader's cursor.
        """
        path = _rel(path)
        pieces = split(text)
        with self._write() as conn:
            live = [dict(r) for r in conn.execute(
                "SELECT * FROM chunk WHERE path = ? AND dead = 0 ORDER BY ord",
                (path,))]
            keep: dict[int, dict] = {}
            matcher = SequenceMatcher(None, [r["hash"] for r in live],
                                      [p.hash for p in pieces], autojunk=False)
            for tag, i1, i2, j1, _j2 in matcher.get_opcodes():
                if tag == "equal":
                    for offset in range(i2 - i1):
                        keep[j1 + offset] = live[i1 + offset]
            gone = [r for r in live if r["id"] not in {k["id"] for k in keep.values()}]
            fresh = [(i, p) for i, p in enumerate(pieces) if i not in keep]
            result = Sync(path=path, added=len(fresh), removed=len(gone),
                          kept=len(keep))
            if not fresh and not gone:
                # Unchanged content, but the surrounding file may have moved
                # around it; keeping the line numbers honest is free.
                self._reposition(conn, keep, pieces)
                return result
            result.rev = self._bump(conn)
            now = time.time()
            for row in gone:
                # The body goes: a dead chunk exists to tell a reader the text
                # is no longer there, and keeping a copy of it to say so would
                # grow the database by every edit ever made.
                conn.execute("UPDATE chunk SET dead = ?, body = '' WHERE id = ?",
                             (result.rev, row["id"]))
                conn.execute("DELETE FROM search WHERE ref = ?",
                             (f"chunk:{row['id']}",))
            self._reposition(conn, keep, pieces)
            for index, piece in fresh:
                cursor = conn.execute(
                    "INSERT INTO chunk(path, ord, hash, heading, first_line,"
                    " last_line, body, rev, at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (path, index, piece.hash, piece.heading, piece.first_line,
                     piece.last_line, piece.body, result.rev, now))
                self._index(conn, "chunk", f"chunk:{cursor.lastrowid}", path,
                            piece.heading, piece.body)
        self.log.debug("context.sync", path=path, rev=result.rev,
                       added=result.added, removed=result.removed)
        return result

    @staticmethod
    def _reposition(conn: sqlite3.Connection, keep: dict[int, dict],
                    pieces: list["Piece"]) -> None:
        """Move surviving chunks to where they now sit, without a new revision.

        Their text is what a reader holds, and it has not changed; a delta
        that repeated them because a line number moved would be the whole
        file again.
        """
        for index, row in keep.items():
            piece = pieces[index]
            if (row["ord"], row["first_line"]) == (index, piece.first_line):
                continue
            conn.execute("UPDATE chunk SET ord = ?, first_line = ?, last_line = ? "
                         "WHERE id = ?",
                         (index, piece.first_line, piece.last_line, row["id"]))

    def forget_file(self, path: str) -> Sync:
        """Record that a file is gone, so readers holding it are told."""
        path = _rel(path)
        with self._write() as conn:
            live = conn.execute("SELECT id FROM chunk WHERE path = ? AND dead = 0",
                                (path,)).fetchall()
            if not live:
                return Sync(path=path)
            rev = self._bump(conn)
            for row in live:
                conn.execute("UPDATE chunk SET dead = ?, body = '' WHERE id = ?",
                             (rev, row["id"]))
                conn.execute("DELETE FROM search WHERE ref = ?", (f"chunk:{row['id']}",))
            return Sync(path=path, rev=rev, removed=len(live))

    def sync_paths(self, root: Path, paths: Iterable[str]) -> list[Sync]:
        """Ingest named files from disk, skipping what is not worth storing."""
        root = Path(root)
        out: list[Sync] = []
        for name in dict.fromkeys(_rel(p) for p in paths if str(p).strip()):
            target = root / name
            if not target.is_file():
                out.append(self.forget_file(name))
                continue
            try:
                if target.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = target.read_text(errors="replace")
            except OSError:
                continue
            if "\x00" in text[:4096]:
                continue  # binary: nothing here an agent would read
            out.append(self.sync_file(name, text))
        return out

    def sync_tree(self, root: Path) -> list[Sync]:
        """Ingest the whole project. This is also how a lost database is rebuilt."""
        root = Path(root)
        found: list[str] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or any(part in SKIP for part in path.parts):
                continue
            found.append(str(path.relative_to(root)))
        synced = self.sync_paths(root, found)
        known = {r["path"] for r in self._conn().execute(
            "SELECT DISTINCT path FROM chunk WHERE dead = 0")}
        for missing in sorted(known - set(found)):
            synced.append(self.forget_file(missing))
        return synced

    # -- reading files -----------------------------------------------------
    def file(self, path: str) -> str:
        """The file as stored, reassembled from its live chunks."""
        rows = self._conn().execute(
            "SELECT body FROM chunk WHERE path = ? AND dead = 0 ORDER BY ord",
            (_rel(path),)).fetchall()
        return "\n".join(r["body"] for r in rows)

    def outline(self, path: str) -> list[Chunk]:
        """A file's chunks without their bodies: its table of contents.

        What an agent deciding *whether* to read something needs, at a
        hundredth of the cost of reading it.
        """
        return [_chunk(r, body=False) for r in self._conn().execute(
            "SELECT path, ord, hash, heading, first_line, last_line, rev, at "
            "FROM chunk WHERE path = ? AND dead = 0 ORDER BY ord", (_rel(path),))]

    def brief(self, paths: Iterable[str], budget: int = 2000) -> str:
        """Outlines of several files, as a prompt block."""
        lines: list[str] = []
        used = 0
        for path in paths:
            chunks = self.outline(path)
            if not chunks:
                continue
            head = f"{path} ({len(chunks)} sections, {chunks[-1].last_line} lines)"
            body = "\n".join(f"  {c.first_line:>5}: {c.heading}" for c in chunks)
            if used + len(head) + len(body) > budget:
                lines.append("… (more files not listed)")
                break
            lines.append(f"{head}\n{body}")
            used += len(head) + len(body) + 2
        return "\n".join(lines)

    def paths(self) -> list[str]:
        return [r["path"] for r in self._conn().execute(
            "SELECT DISTINCT path FROM chunk WHERE dead = 0 ORDER BY path")]

    # -- catching up --------------------------------------------------------
    def delta(self, since: int = 0, *, paths: Iterable[str] = (),
              kinds: Iterable[str] = (), limit: int = 200) -> Delta:
        """Everything recorded after revision *since*.

        *paths* narrows it to work touching those files, which is what a
        ticket-scoped reader wants: the rest of the project moved, but not
        under it.

        Oldest revision first, in both tables and in the `limit`. That is what
        makes a reader on a small budget converge: :meth:`Delta.render` stops
        the cursor at the first thing it could not fit, so filling from the
        oldest end means everything before that point is genuinely seen and
        never offered again. Newest-first would spend the same budget on the
        newest changes and leave the cursor where it was, and the reader would
        be handed them again on every call.
        """
        conn = self._conn()
        since = max(0, int(since))
        result = Delta(since=since, head=self.head)
        if result.head <= since:
            return result
        wanted = [_rel(p) for p in paths]
        args: list[Any] = [since]
        sql = ["SELECT r.* FROM record r"]
        if wanted:
            sql.append("JOIN record_path p ON p.record = r.id AND p.path IN "
                       f"({','.join('?' * len(wanted))})")
            args = wanted + args
        sql.append("WHERE r.rev > ?")
        if kinds:
            kinds = list(kinds)
            sql.append(f"AND r.kind IN ({','.join('?' * len(kinds))})")
            args.extend(kinds)
        sql.append("GROUP BY r.id ORDER BY r.rev LIMIT ?")
        args.append(int(limit))
        result.records = [_record(r) for r in conn.execute(" ".join(sql), args)]

        chunk_sql = ["SELECT * FROM chunk WHERE dead = 0 AND rev > ?"]
        chunk_args: list[Any] = [since]
        if wanted:
            chunk_sql.append(f"AND path IN ({','.join('?' * len(wanted))})")
            chunk_args.extend(wanted)
        chunk_sql.append("ORDER BY rev, path, ord LIMIT ?")
        chunk_args.append(int(limit))
        result.chunks = [_chunk(r) for r in conn.execute(" ".join(chunk_sql), chunk_args)]

        dead_sql = ["SELECT path, MAX(dead) AS rev FROM chunk WHERE dead > ?"]
        dead_args: list[Any] = [since]
        if wanted:
            dead_sql.append(f"AND path IN ({','.join('?' * len(wanted))})")
            dead_args.extend(wanted)
        dead_sql.append("GROUP BY path")
        alive = set(self.paths())
        # A file whose chunks died because they were rewritten is not gone; it
        # is in the changed list above. Only one with nothing live left is.
        result.removed = [Gone(r["path"], int(r["rev"]))
                          for r in conn.execute(" ".join(dead_sql), dead_args)
                          if r["path"] not in alive]
        return result

    def catch_up(self, reader: str, *, paths: Iterable[str] = (),
                 kinds: Iterable[str] = (), budget: int = 4000,
                 advance: bool = True) -> Delta:
        """Update *reader* from wherever it left off, and move its cursor.

        This is the call an agent makes instead of re-reading the project. A
        reader that has never been here has cursor 0 and is handed the history
        it would have read anyway, so the first call and the tenth are the
        same call.

        The cursor advances only as far as the returned text actually goes:
        what the budget squeezed out is still unseen, and will come back in
        the next update rather than being quietly skipped.
        """
        cursor = self.cursor(reader)
        delta = self.delta(cursor, paths=paths, kinds=kinds)
        delta.render(budget)
        if advance and delta.through > cursor:
            self.set_cursor(reader, delta.through)
        self.log.debug("context.catch_up", reader=reader, since=cursor,
                       through=delta.through, records=len(delta.records),
                       chunks=len(delta.chunks), dropped=delta.dropped)
        return delta

    def cursor(self, reader: str) -> int:
        row = self._conn().execute("SELECT rev FROM cursor WHERE name = ?",
                                   (str(reader),)).fetchone()
        return int(row["rev"]) if row else 0

    def set_cursor(self, reader: str, rev: int) -> int:
        rev = max(0, int(rev))
        with self._write() as conn:
            conn.execute(
                "INSERT INTO cursor(name, rev, updated, reads) VALUES(?,?,?,1) "
                "ON CONFLICT(name) DO UPDATE SET rev = excluded.rev, "
                "updated = excluded.updated, reads = reads + 1",
                (str(reader), rev, time.time()))
        return rev

    def cursors(self) -> dict[str, int]:
        return {r["name"]: int(r["rev"]) for r in
                self._conn().execute("SELECT name, rev FROM cursor ORDER BY name")}

    # -- search ------------------------------------------------------------
    def search(self, query: str, *, limit: int = 10, kinds: Iterable[str] = (),
               path: str = "") -> list[Hit]:
        """Find records and code by words, best first.

        FTS5 where the build has it, a scan where it does not. The query is
        reduced to bare terms rather than passed through, because it arrives
        from a model that will eventually write `parse(` in it and an
        unparseable query should return nothing found, not an exception.
        """
        terms = [t for t in re.findall(r"[\w.]{2,}", str(query))][:12]
        if not terms:
            return []
        kinds = list(kinds)
        if self.fts:
            try:
                return self._search_fts(terms, limit, kinds, path)
            except sqlite3.OperationalError as exc:  # pragma: no cover - build-specific
                self.log.warn("context.search_failed", error=str(exc)[:120])
        return self._search_scan(terms, limit, kinds, path)

    def _search_fts(self, terms: list[str], limit: int, kinds: list[str],
                    path: str) -> list[Hit]:
        match = " OR ".join(f'"{t}"' for t in terms)
        sql = ["SELECT kind, ref, path, title, snippet(search, 4, '', '', ' … ', 24)"
               " AS body FROM search WHERE search MATCH ?"]
        args: list[Any] = [match]
        if kinds:
            sql.append(f"AND kind IN ({','.join('?' * len(kinds))})")
            args.extend(kinds)
        if path:
            sql.append("AND path = ?")
            args.append(_rel(path))
        sql.append("ORDER BY rank LIMIT ?")
        args.append(int(limit))
        return [Hit(kind=r["kind"], ref=r["ref"], path=r["path"], title=r["title"],
                    body=r["body"]) for r in self._conn().execute(" ".join(sql), args)]

    def _search_scan(self, terms: list[str], limit: int, kinds: list[str],
                     path: str) -> list[Hit]:
        """The no-FTS5 fallback: rank by how many terms a row contains."""
        lowered = [t.lower() for t in terms]
        hits: list[tuple[int, Hit]] = []
        conn = self._conn()
        if not kinds or "chunk" not in kinds:
            for row in conn.execute("SELECT * FROM record ORDER BY rev DESC LIMIT 2000"):
                if kinds and row["kind"] not in kinds:
                    continue
                text = f"{row['title']}\n{row['body']}".lower()
                score = sum(t in text for t in lowered)
                if score:
                    hits.append((score, Hit(kind=row["kind"], ref=f"record:{row['id']}",
                                            path=(json.loads(row["paths"]) or [""])[0],
                                            title=row["title"],
                                            body=row["body"][:400], rev=row["rev"])))
        if not kinds or "chunk" in kinds:
            sql = "SELECT * FROM chunk WHERE dead = 0"
            args: list[Any] = []
            if path:
                sql += " AND path = ?"
                args.append(_rel(path))
            for row in conn.execute(sql + " ORDER BY rev DESC LIMIT 4000", args):
                text = row["body"].lower()
                score = sum(t in text for t in lowered)
                if score:
                    hits.append((score, Hit(kind="chunk", ref=f"chunk:{row['id']}",
                                            path=row["path"], title=row["heading"],
                                            body=row["body"][:400], rev=row["rev"])))
        hits.sort(key=lambda pair: (-pair[0], -pair[1].rev))
        return [hit for _score, hit in hits[:limit]]

    @staticmethod
    def _index(conn: sqlite3.Connection, kind: str, ref: str, path: str,
               title: str, body: str) -> None:
        try:
            conn.execute("DELETE FROM search WHERE ref = ?", (ref,))
            conn.execute("INSERT INTO search(kind, ref, path, title, body) "
                         "VALUES(?,?,?,?,?)", (kind, ref, path, title, body))
        except sqlite3.OperationalError:
            pass  # no FTS5 in this build; _search_scan covers it

    # -- housekeeping -------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        conn = self._conn()
        def count(sql: str, *args: Any) -> int:
            return int(conn.execute(sql, args).fetchone()[0])
        return {
            "head": self.head,
            "records": count("SELECT COUNT(*) FROM record"),
            "facts": count("SELECT COUNT(*) FROM record WHERE key <> '' AND stale = 0"),
            "files": count("SELECT COUNT(DISTINCT path) FROM chunk WHERE dead = 0"),
            "chunks": count("SELECT COUNT(*) FROM chunk WHERE dead = 0"),
            "buried": count("SELECT COUNT(*) FROM chunk WHERE dead > 0"),
            "readers": count("SELECT COUNT(*) FROM cursor"),
            "fts": self.fts,
            "bytes": self.path.stat().st_size if self.path.exists() else 0,
        }

    def prune(self, keep_records: int = 5000) -> dict[str, int]:
        """Drop what no reader can still be owed.

        A dead chunk matters only until every cursor is past the revision that
        killed it; after that it is history nobody will be told. Records are
        kept far longer, and keyed ones forever: they are the cheap rows, and
        they are the ones that answer why something is the way it is.
        """
        floor = min(self.cursors().values(), default=self.head)
        with self._write() as conn:
            dead = conn.execute("DELETE FROM chunk WHERE dead > 0 AND dead <= ?",
                                (floor,)).rowcount
            old = conn.execute(
                "DELETE FROM record WHERE key = '' AND kind = 'log' AND id NOT IN "
                "(SELECT id FROM record ORDER BY rev DESC LIMIT ?)",
                (int(keep_records),)).rowcount
            conn.execute("DELETE FROM record_path WHERE record NOT IN "
                         "(SELECT id FROM record)")
            try:
                conn.execute("DELETE FROM search WHERE ref LIKE 'record:%' AND "
                             "CAST(SUBSTR(ref, 8) AS INTEGER) NOT IN "
                             "(SELECT id FROM record)")
            except sqlite3.OperationalError:
                pass
        self.log.info("context.prune", chunks=dead, records=old)
        return {"chunks": dead, "records": old}


# -- chunking ----------------------------------------------------------------


@dataclass
class Piece:
    """A candidate chunk, before it meets what is already stored."""

    body: str
    first_line: int
    last_line: int

    @property
    def hash(self) -> str:
        return hashlib.blake2b(self.body.encode("utf-8", "replace"),
                               digest_size=8).hexdigest()

    @property
    def heading(self) -> str:
        for line in self.body.splitlines():
            if line.strip():
                return line.strip()[:120]
        return f"lines {self.first_line}-{self.last_line}"


def split(text: str) -> list[Piece]:
    """Cut *text* where it already divides itself.

    Every cut is decided by looking at one line and the line above it, and
    never by how much has been read so far. That locality is the whole
    property: inserting a function at the top of a file leaves every boundary
    below it exactly where it was, so those chunks keep their hashes and a
    reader holding them is sent only the new one. A fixed window, or any rule
    that counts lines since the last cut, would shift every boundary after
    the insertion and make a one-line edit cost the whole file -- which is
    the outcome this module exists to avoid.
    """
    if not text:
        return []
    # ``split`` rather than ``splitlines``: the pieces have to rejoin into the
    # exact bytes that came in, or :meth:`ContextDB.file` would hand back a
    # file that differs from the one on disk by its final newline -- and an
    # agent editing what it was handed would write that difference back.
    lines = text.split("\n")
    numbered = len(lines) - 1 if lines[-1] == "" else len(lines)
    starts = [0] + [i for i in range(1, len(lines)) if _is_boundary(lines, i)]
    pieces: list[Piece] = []
    for first, stop in zip(starts, starts[1:] + [len(lines)]):
        # Long stretches with no boundary in them are cut to size. The cut is
        # made inside one stretch and nowhere else, so it cannot renumber the
        # chunks on either side of it.
        cursor = first
        while cursor < stop:
            end, size = cursor, 0
            while end < stop and end - cursor < MAX_LINES and size < MAX_CHARS:
                size += len(lines[end]) + 1
                end += 1
            pieces.append(Piece("\n".join(lines[cursor:end]), cursor + 1,
                                min(end, numbered)))
            cursor = end
    return pieces


def _is_boundary(lines: list[str], index: int) -> bool:
    line = lines[index]
    if not line.strip() or line[:1].isspace():
        return False  # indented: part of whatever contains it
    return not lines[index - 1].strip() or bool(_BOUNDARY.match(line))


# -- helpers -----------------------------------------------------------------


def open_db(root: Path, log: Logger = NULL) -> ContextDB:
    """The context database for a workspace, created if it is not there yet."""
    return ContextDB.for_workspace(root, log)


def _section(lines: list[str], header: str, items: list, join: str = "\n\n") -> None:
    """Append one titled section to *lines*, or nothing when it is empty."""
    if not items:
        return
    lines.append(header)
    lines.append(join.join(item.render() for item in items))


def _record(row: sqlite3.Row) -> Record:
    return Record(id=row["id"], rev=row["rev"], at=row["at"], kind=row["kind"],
                  agent=row["agent"], ticket=row["ticket"], title=row["title"],
                  body=row["body"], key=row["key"], weight=row["weight"],
                  paths=json.loads(row["paths"] or "[]"),
                  meta=json.loads(row["meta"] or "{}"))


def _chunk(row: sqlite3.Row, body: bool = True) -> Chunk:
    return Chunk(path=row["path"], ord=row["ord"], hash=row["hash"],
                 heading=row["heading"], first_line=row["first_line"],
                 last_line=row["last_line"], rev=row["rev"], at=row["at"],
                 body=row["body"] if body else "")


def _rel(path: Any) -> str:
    """A workspace-relative path, in the one spelling the tables index by."""
    text = str(path).strip().replace("\\", "/").lstrip("/")
    parts = [p for p in text.split("/") if p not in ("", ".")]
    return "/".join(parts)


def _fields(detail: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in detail.items())


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())
