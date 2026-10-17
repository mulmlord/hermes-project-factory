"""Git automation and optional GitHub publishing.

Credentials are never written into a generated project: GitHub work goes
through :mod:`hermes.github`, which reads the operator's token on the host and
hands it to nothing but GitHub's API and a single ``git push``.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from . import github as api
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
    """Publishing through GitHub's REST API. Every method is a no-op when disabled."""

    def __init__(self, cfg: dict, log: Logger = NULL, client: api.Client | None = None):
        self.cfg = cfg or {}
        self.log = log
        self.enabled = bool(self.cfg.get("enabled"))
        self.token = api.find_token(self.cfg)
        self.web = str(self.cfg.get("web_url") or api.WEB)
        self.client = client or api.Client(self.token, str(self.cfg.get("api_url") or api.API))
        self._ready: bool | None = None

    def ready(self) -> bool:
        if not self.enabled:
            return False
        if self._ready is not None:
            return self._ready
        if not self.client.token:
            self.log.warn("github.no_token",
                          hint="set HERMES_GITHUB_TOKEN or github.token_file")
            self._ready = False
            return False
        try:
            self.client.login()
            self._ready = True
        except api.GitHubError as exc:
            self.log.warn("github.not_authenticated", error=str(exc)[:200])
            self._ready = False
        return self._ready

    def _slug(self, name: str) -> api.Slug:
        return api.Slug(self.cfg.get("owner") or self.client.login(), name)

    def create(self, root: Path, name: str, description: str) -> str | None:
        """Create the repository and wire up the remote, pushing nothing.

        Creation and push answer different questions, so they are separate
        calls. An issue cannot be opened against a repository that does not
        exist, and the board is worth watching from the moment it exists; the
        code is worth showing only once a ticket has passed its gates.
        Creating early and pushing late is what makes a running project
        observable without publishing a scaffold nobody has validated.
        """
        if not self.ready():
            return None
        slug = self._slug(name)
        try:
            existing = self.client.repository(slug)
            if existing is None:
                existing = self.client.create_repository(
                    slug.name, private=bool(self.cfg.get("private", True)),
                    description=description, org=slug.owner)
                self.log.info("github.created", repo=str(slug), pushed=False)
        except api.GitHubError as exc:
            self.log.warn("github.create_failed", error=str(exc)[:300])
            return None
        url = str(existing.get("html_url") or f"{self.web}/{slug}")
        self._ensure_remote(root, url)
        return url

    def _ensure_remote(self, root: Path, url: str) -> None:
        """Point origin at an existing repository, if it is not already."""
        if subprocess.run(["git", "remote", "get-url", "origin"], cwd=root,
                          capture_output=True, text=True).returncode == 0:
            return
        subprocess.run(["git", "remote", "add", "origin", f"{url}.git"], cwd=root,
                       capture_output=True, text=True)

    def push(self, root: Path) -> bool:
        if not self.ready():
            return False
        proc = api.push(root, "origin", "HEAD", self.token, web=self.web, extra=("-u",))
        if proc.returncode != 0:
            self.log.warn("github.push_failed", error=proc.stderr.strip()[:300])
            return False
        self.log.info("github.pushed")
        return True

    def create_and_push(self, root: Path, name: str, description: str) -> str | None:
        """Publish the repository and its code in one step, as release does.

        Nothing is pulled back: the board is the source of truth, and §3
        settled that an edit made on GitHub is never read into the project.
        """
        url = self.create(root, name, description)
        if url:
            self.push(root)
        return url

    def release(self, root: Path, tag: str, title: str, notes: str) -> str | None:
        if not self.ready():
            return None
        slug = api.remote_slug(root)
        if slug is None:
            self.log.warn("github.release_failed", error="origin is not a GitHub remote")
            return None
        api.push(root, "origin", f"refs/tags/{tag}", self.token, web=self.web)
        try:
            return self.client.release(slug, tag, title, notes)
        except api.GitHubError as exc:
            self.log.warn("github.release_failed", error=str(exc)[:200])
            return None


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
        self._remote: api.Slug | None = None

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

    def _repository(self) -> api.Slug | None:
        if self._remote is None:
            self._remote = api.remote_slug(self.root)
        return self._remote

    def _open(self, title: str, body: str) -> int | None:
        slug = self._repository()
        if slug is None:
            self.log.warn("github.issue_create_failed", title=title[:60],
                          error="origin is not a GitHub remote")
            return None
        try:
            number = self.github.client.open_issue(slug, title, body)
        except api.GitHubError as exc:
            self.log.warn("github.issue_create_failed", title=title[:60],
                          error=str(exc)[:200])
            return None
        self.log.info("github.issue_opened", number=number, title=title[:60])
        return number

    def _edit(self, number: int, title: str, body: str, closed: bool,
              was_closed: bool) -> bool:
        slug = self._repository()
        if slug is None:
            return False
        fields = {"title": title, "body": body}
        if closed != was_closed:
            fields["state"] = "closed" if closed else "open"
        try:
            self.github.client.edit_issue(slug, number, **fields)
        except api.GitHubError as exc:
            self.log.warn("github.issue_edit_failed", number=number,
                          error=str(exc)[:200])
            return False
        if closed != was_closed:
            self.log.info("github.issue_state", number=number,
                          state="closed" if closed else "reopened")
        return True


def _split_author(author: str) -> tuple[str, str]:
    if "<" in author and ">" in author:
        name, _, rest = author.partition("<")
        return name.strip() or "Hermes Factory", rest.rstrip("> ").strip()
    return author.strip() or "Hermes Factory", "hermes@localhost"
