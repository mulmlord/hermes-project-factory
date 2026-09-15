"""Git automation and optional GitHub publishing.

Credentials are never written into a generated project: GitHub work goes
through the ``gh`` CLI, which holds its own auth on the host.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from .log import Logger, NULL
from .tickets import FINISHED

PREFIXES = ("feat", "fix", "test", "perf", "docs", "refactor", "chore", "build")


class GitError(RuntimeError):
    pass


class Repo:
    def __init__(self, root: Path, log: Logger = NULL,
                 author: str = "Hermes Factory <hermes@localhost>"):
        self.root = Path(root)
        self.log = log
        self.name, self.email = _split_author(author)
        self.available = bool(shutil.which("git"))

    def _run(self, *args: str, check: bool = False) -> subprocess.CompletedProcess:
        proc = subprocess.run(["git", *args], cwd=self.root, capture_output=True,
                              text=True)
        if check and proc.returncode != 0:
            raise GitError(f"git {' '.join(args)}: {proc.stderr.strip()}")
        return proc

    def init(self) -> bool:
        if not self.available:
            self.log.warn("git.unavailable")
            return False
        if (self.root / ".git").is_dir():
            return True
        self._run("init", "-q", check=True)
        self._run("config", "user.name", self.name)
        self._run("config", "user.email", self.email)
        self._run("checkout", "-q", "-b", "main")
        self.log.info("git.init", root=str(self.root))
        return True

    def commit(self, message: str, *, kind: str = "chore") -> str | None:
        """Stage everything and commit. Returns the short SHA, or None if clean."""
        if not self.available or not (self.root / ".git").is_dir():
            return None
        self._run("add", "-A")
        if not self._run("diff", "--cached", "--quiet").returncode:
            return None  # nothing staged
        prefix = kind if kind in PREFIXES else "chore"
        subject = message.strip().splitlines()[0][:72]
        if not subject.startswith(tuple(f"{p}:" for p in PREFIXES)):
            subject = f"{prefix}: {subject}"
        body = "\n".join(message.strip().splitlines()[1:]).strip()
        full = subject + (f"\n\n{body}" if body else "")
        proc = self._run("commit", "-q", "-m", full)
        if proc.returncode != 0:
            self.log.warn("git.commit_failed", error=proc.stderr.strip()[:200])
            return None
        sha = self._run("rev-parse", "--short", "HEAD").stdout.strip()
        self.log.info("git.commit", sha=sha, subject=subject)
        return sha

    def files_in(self, ref: str = "HEAD") -> list[str]:
        """Paths a commit touched: what an agent changed, as opposed to claimed."""
        if not self.available or not (self.root / ".git").is_dir():
            return []
        out = self._run("show", "--name-only", "--pretty=format:", ref).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]

    def branch(self, name: str) -> None:
        self._run("checkout", "-q", "-B", name, check=True)

    def current_branch(self) -> str:
        return self._run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    def tag(self, name: str, message: str = "") -> None:
        self._run("tag", "-f", "-a", name, "-m", message or name, check=True)
        self.log.info("git.tag", tag=name)

    def rollback(self, sha: str = "HEAD~1") -> None:
        self._run("reset", "--hard", sha, check=True)
        self.log.warn("git.rollback", to=sha)

    def head(self) -> str:
        return self._run("rev-parse", "HEAD").stdout.strip()

    def diff(self, ref: str = "HEAD", limit: int = 20000) -> str:
        return self._run("diff", ref).stdout[:limit]

    def log_lines(self, count: int = 20) -> list[str]:
        out = self._run("log", f"-{count}", "--oneline").stdout
        return out.strip().splitlines()


class GitHub:
    """Thin wrapper over the ``gh`` CLI. Every method is a no-op when disabled."""

    def __init__(self, cfg: dict, log: Logger = NULL):
        self.cfg = cfg or {}
        self.log = log
        self.enabled = bool(self.cfg.get("enabled"))
        self.available = bool(shutil.which("gh"))

    def _gh(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["gh", *args], cwd=cwd, capture_output=True, text=True)

    def ready(self) -> bool:
        if not self.enabled:
            return False
        if not self.available:
            self.log.warn("github.gh_missing")
            return False
        if self._gh("auth", "status").returncode != 0:
            self.log.warn("github.not_authenticated")
            return False
        return True

    def create_and_push(self, root: Path, name: str, description: str) -> str | None:
        if not self.ready():
            return None
        visibility = "--private" if self.cfg.get("private", True) else "--public"
        owner = self.cfg.get("owner")
        slug = f"{owner}/{name}" if owner else name
        existing = self._gh("repo", "view", slug, "--json", "url", cwd=root)
        if existing.returncode == 0:
            url = json.loads(existing.stdout).get("url")
            self._gh("repo", "sync", cwd=root)
            push = subprocess.run(["git", "push", "-u", "origin", "HEAD"], cwd=root,
                                  capture_output=True, text=True)
            self.log.info("github.pushed", repo=slug, ok=push.returncode == 0)
            return url
        proc = self._gh("repo", "create", slug, visibility, "--source", ".",
                        "--remote", "origin", "--push",
                        "--description", description[:250], cwd=root)
        if proc.returncode != 0:
            self.log.warn("github.create_failed", error=proc.stderr.strip()[:300])
            return None
        view = self._gh("repo", "view", slug, "--json", "url", cwd=root)
        url = json.loads(view.stdout).get("url") if view.returncode == 0 else None
        self.log.info("github.created", repo=slug, url=url)
        return url

    def release(self, root: Path, tag: str, title: str, notes: str) -> str | None:
        if not self.ready():
            return None
        subprocess.run(["git", "push", "--tags"], cwd=root, capture_output=True)
        proc = self._gh("release", "create", tag, "--title", title, "--notes", notes[:60000],
                        cwd=root)
        if proc.returncode != 0:
            self.log.warn("github.release_failed", error=proc.stderr.strip()[:200])
            return None
        return proc.stdout.strip()


ONE_WAY = (
    "\n\n---\n*Projected from the Hermes ticket board. The board is the source "
    "of truth and this projection is one way: an edit made here is not read "
    "back, because the factory is specified to run without a human in the "
    "loop. Change the project by changing the board.*\n"
)


def issue_body(ticket) -> str:
    """A ticket rendered as an issue: everything an onlooker needs, no internals."""
    def block(name: str, items) -> str:
        return f"### {name}\n" + ("\n".join(f"- {i}" for i in items) or "- (none)") + "\n\n"

    body = (
        f"**Status:** {ticket.status} · **Kind:** {ticket.kind} · "
        f"**Priority:** {ticket.priority} · **Risk:** {ticket.risk} · "
        f"**Attempts:** {ticket.attempts}\n\n"
        f"### Objective\n{ticket.objective or ticket.title}\n\n"
        + (f"### Context\n{ticket.context}\n\n" if ticket.context else "")
        + block("Acceptance criteria", ticket.acceptance)
        + block("Validation", [f"`{v}`" for v in ticket.validation])
        + block("Expected artifacts", ticket.artifacts)
        + (block("Depends on", ticket.dependencies) if ticket.dependencies else "")
    )
    return body[:60000] + ONE_WAY


class Issues:
    """One-way projection of the ticket board onto GitHub issues.

    Section 23 asks for issue synchronisation; this is the half of it that
    does not put a human back into an unattended loop. Nothing is ever read
    from GitHub, and every issue says so in its own footer.

    Only a ticket whose issue would actually differ costs an API call, so the
    save that runs every iteration is free on an unchanged board, and free
    entirely until the project has a repository to synchronise with.
    """

    #: How long a failed readiness check is trusted before being retried. The
    #: repository only appears at release, so "not ready" is usually temporary.
    RECHECK = 60.0

    def __init__(self, github: GitHub, root: Path, log: Logger = NULL,
                 batch: int = 20):
        self.github = github
        self.root = Path(root)
        self.log = log
        self.batch = batch
        self.path = self.root / ".hermes" / "state" / "github_issues.json"
        self.known: dict[str, dict] = self._load()
        self._ready = False
        self._checked = 0.0

    def _load(self) -> dict:
        if not self.path.is_file():
            return {}
        try:
            return json.loads(self.path.read_text()).get("issues", {})
        except (json.JSONDecodeError, OSError):
            # A lost map means duplicate issues, which is worth a warning but
            # not worth failing a build over.
            self.log.warn("github.issue_map_unreadable", path=str(self.path))
            return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"issues": self.known}, indent=2))

    def ready(self) -> bool:
        """True once GitHub is usable and the project has a remote.

        Before publication there is no repository, so this is what keeps an
        ordinary run from making any network call at all.
        """
        if self._ready:
            return True
        now = time.time()
        if now - self._checked < self.RECHECK:
            return False
        self._checked = now
        if not self.github.ready():
            return False
        remote = subprocess.run(["git", "remote", "get-url", "origin"],
                                cwd=self.root, capture_output=True, text=True)
        self._ready = remote.returncode == 0
        return self._ready

    def sync(self, tickets) -> int:
        """Bring issues in line with the board. Returns the API calls spent."""
        if not self.ready():
            return 0
        spent, dirty = 0, False
        for ticket in sorted(tickets, key=lambda t: t.id):
            if spent >= self.batch:
                # The rest is picked up next iteration: a freshly planned
                # board should not spend a hundred API calls in one save.
                self.log.info("github.issues_deferred", synced=spent)
                break
            called, changed = self._project(ticket)
            spent += int(called)
            dirty = dirty or changed
        if dirty:
            self._save()
        return spent

    def _project(self, ticket) -> tuple[bool, bool]:
        """One ticket. Returns (an API call was made, the map changed)."""
        title = f"{ticket.id}: {ticket.title}"[:250]
        body = issue_body(ticket)
        closed = ticket.status in FINISHED
        digest = hashlib.sha256(f"{title}\n{body}\n{closed}".encode()).hexdigest()[:16]
        record = self.known.get(ticket.id, {})
        if record.get("digest") == digest:
            return False, False

        number = record.get("number")
        if number is None:
            if closed:
                # Finished before the repository existed: opening an issue in
                # order to close it is noise, not observability.
                self.known[ticket.id] = {"number": None, "digest": digest,
                                         "closed": True}
                return False, True
            number = self._open(title, body)
            if number is None:
                return False, False
            self.known[ticket.id] = {"number": number, "digest": digest,
                                     "closed": False}
            return True, True

        if not self._edit(number, title, body, closed, bool(record.get("closed"))):
            return False, False
        self.known[ticket.id] = {"number": number, "digest": digest, "closed": closed}
        return True, True

    def _open(self, title: str, body: str) -> int | None:
        proc = self.github._gh("issue", "create", "--title", title, "--body", body,
                               cwd=self.root)
        if proc.returncode != 0:
            self.log.warn("github.issue_create_failed", title=title[:60],
                          error=proc.stderr.strip()[:200])
            return None
        number = _issue_number(proc.stdout)
        self.log.info("github.issue_opened", number=number, title=title[:60])
        return number

    def _edit(self, number: int, title: str, body: str, closed: bool,
              was_closed: bool) -> bool:
        proc = self.github._gh("issue", "edit", str(number), "--title", title,
                               "--body", body, cwd=self.root)
        if proc.returncode != 0:
            self.log.warn("github.issue_edit_failed", number=number,
                          error=proc.stderr.strip()[:200])
            return False
        if closed != was_closed:
            self.github._gh("issue", "close" if closed else "reopen", str(number),
                            cwd=self.root)
            self.log.info("github.issue_state", number=number,
                          state="closed" if closed else "reopened")
        return True


def _issue_number(stdout: str) -> int | None:
    """``gh issue create`` prints the URL of what it made."""
    match = re.search(r"/issues/(\d+)", stdout or "")
    return int(match.group(1)) if match else None


def _split_author(author: str) -> tuple[str, str]:
    if "<" in author and ">" in author:
        name, _, rest = author.partition("<")
        return name.strip() or "Hermes Factory", rest.rstrip("> ").strip()
    return author.strip() or "Hermes Factory", "hermes@localhost"
