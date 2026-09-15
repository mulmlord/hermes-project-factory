"""Git automation and optional GitHub publishing.

Credentials are never written into a generated project: GitHub work goes
through the ``gh`` CLI, which holds its own auth on the host.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .log import Logger, NULL

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


def _split_author(author: str) -> tuple[str, str]:
    if "<" in author and ">" in author:
        name, _, rest = author.partition("<")
        return name.strip() or "Hermes Factory", rest.rstrip("> ").strip()
    return author.strip() or "Hermes Factory", "hermes@localhost"
