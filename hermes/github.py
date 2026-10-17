"""GitHub's REST API, spoken directly instead of through the ``gh`` CLI.

Hermes published through ``gh``, a separate program holding its own login.
What the factory asks of GitHub is a short list -- who am I, does this
repository exist, create it, fork it, open an issue or edit one, open a pull
request, cut a release -- and every item is one HTTPS request with a JSON body.
This module makes those requests with :mod:`urllib`, and pushes with ``git``,
which the factory already requires.

The token is the operator's, read from the environment
(``HERMES_GITHUB_TOKEN``, ``GITHUB_TOKEN`` or ``GH_TOKEN``) or from the file
``github.token_file`` names, ``~/.config/hermes/github-token`` by default. It is never written anywhere: not into a
generated project, not into a remote URL, not into ``git``'s argv where any
process on the machine can read it. A push gets it through ``git``'s
environment-only configuration (``GIT_CONFIG_COUNT``), as an HTTP header
scoped to the GitHub host.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sandbox import carries_secret

API = "https://api.github.com"
WEB = "https://github.com"
TOKEN_VARIABLES = ("HERMES_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")


class GitHubError(RuntimeError):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


def find_token(cfg: dict | None = None) -> str:
    """The operator's token, or ``""``. Never from a workspace."""
    for name in TOKEN_VARIABLES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    path = (cfg or {}).get("token_file") or default_token_file()
    try:
        return Path(os.path.expanduser(str(path))).read_text().strip()
    except OSError:
        return ""


def default_token_file() -> Path:
    """Where the token is looked for when nothing names it: a unit started by
    systemd sees none of the shell's environment, so a file is how a
    scheduled run finds it."""
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "hermes" / "github-token"


@dataclass
class Slug:
    owner: str
    name: str

    def __str__(self) -> str:
        return f"{self.owner}/{self.name}"


_REMOTE = re.compile(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$")


def parse_remote(url: str) -> Slug | None:
    """``owner/name`` from an https or ssh GitHub remote URL."""
    match = _REMOTE.search((url or "").strip())
    return Slug(match.group(1), match.group(2)) if match else None


def remote_slug(root: Path, remote: str = "origin") -> Slug | None:
    proc = subprocess.run(["git", "remote", "get-url", remote], cwd=root,
                          capture_output=True, text=True)
    return parse_remote(proc.stdout) if proc.returncode == 0 else None


def auth_env(token: str, web: str = WEB) -> dict[str, str]:
    """Environment that lets one ``git`` command authenticate to GitHub.

    Configuration passed this way is visible to that process alone, and the
    header is scoped to the GitHub host, so a submodule or a second remote
    elsewhere is never sent it.
    """
    if not token:
        return {"GIT_TERMINAL_PROMPT": "0"}
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return {"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": f"http.{web.rstrip('/')}/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}"}


def push(root: Path, remote: str, refspec: str, token: str = "", *,
         web: str = WEB, extra: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    """``git push`` with the token in the environment rather than the command line.

    With a token, that token is the one credential the push needs, so it is
    the one it gets: every other secret-shaped variable is sieved out the way
    the sandbox sieves them. Without one, ``git`` runs as the operator would
    run it and uses their own credential helper.
    """
    env = None
    if token:
        env = {k: v for k, v in os.environ.items() if not carries_secret(k)}
        env.update(auth_env(token, web))
    return subprocess.run(["git", "push", *extra, remote, refspec], cwd=root,
                          capture_output=True, text=True, env=env)


class Client:
    """The handful of REST calls Hermes makes."""

    def __init__(self, token: str, api: str = API, timeout: float = 30):
        self.token = token
        self.api = api.rstrip("/")
        self.timeout = timeout
        self._login: str | None = None

    def request(self, method: str, path: str, body: Any = None,
                expect: tuple[int, ...] = (200, 201)) -> Any:
        url = path if path.startswith("http") else f"{self.api}{path}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method, headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hermes-factory",
            **({"Content-Type": "application/json"} if data is not None else {}),
            **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
        })
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        except (urllib.error.URLError, OSError) as exc:
            raise GitHubError(f"{method} {path}: {exc}") from exc
        try:
            payload = json.loads(raw.decode() or "null")
        except ValueError:
            payload = raw.decode(errors="replace")
        if status not in expect:
            message = payload.get("message", "") if isinstance(payload, dict) else str(payload)
            raise GitHubError(f"{method} {path}: HTTP {status} {message}"[:500], status)
        return payload

    # -- identity and repositories -------------------------------------------
    def login(self) -> str:
        if self._login is None:
            self._login = str(self.request("GET", "/user").get("login", ""))
        return self._login

    def repository(self, slug: Slug) -> dict | None:
        try:
            return self.request("GET", f"/repos/{slug}")
        except GitHubError as exc:
            if exc.status == 404:
                return None
            raise

    def create_repository(self, name: str, *, private: bool = True,
                          description: str = "", org: str | None = None) -> dict:
        path = f"/orgs/{org}/repos" if org and org != self.login() else "/user/repos"
        return self.request("POST", path, {"name": name, "private": private,
                                           "description": description[:350],
                                           "auto_init": False})

    def fork(self, upstream: Slug, *, wait: float = 60.0) -> Slug:
        """The authenticated user's fork of *upstream*, made if it is missing.

        GitHub creates a fork asynchronously, so this waits until the fork
        answers before handing it to a push.
        """
        made = self.request("POST", f"/repos/{upstream}/forks", {},
                            expect=(200, 201, 202))
        slug = Slug(made["owner"]["login"], made["name"])
        deadline = time.monotonic() + wait
        while self.repository(slug) is None:
            if time.monotonic() > deadline:
                raise GitHubError(f"the fork {slug} did not appear within {wait:.0f}s")
            time.sleep(2)
        return slug

    def can_push(self, slug: Slug) -> bool:
        repo = self.repository(slug) or {}
        return bool((repo.get("permissions") or {}).get("push"))

    # -- issues, pull requests, releases -------------------------------------
    def open_issue(self, slug: Slug, title: str, body: str,
                   labels: list[str] | None = None) -> int:
        made = self.request("POST", f"/repos/{slug}/issues",
                            {"title": title, "body": body,
                             **({"labels": labels} if labels else {})})
        return int(made["number"])

    def edit_issue(self, slug: Slug, number: int, **fields) -> dict:
        return self.request("PATCH", f"/repos/{slug}/issues/{number}", fields)

    def open_pull(self, upstream: Slug, *, title: str, head: str, base: str,
                  body: str, draft: bool = False) -> dict:
        return self.request("POST", f"/repos/{upstream}/pulls",
                            {"title": title, "head": head, "base": base,
                             "body": body, "draft": draft,
                             "maintainer_can_modify": True})

    def pulls_from(self, upstream: Slug, head: str) -> list[dict]:
        """Open pull requests whose head is *head* (``owner:branch``)."""
        query = urllib.parse.urlencode({"state": "open", "head": head})
        return self.request("GET", f"/repos/{upstream}/pulls?{query}") or []

    def release(self, slug: Slug, tag: str, title: str, notes: str) -> str:
        made = self.request("POST", f"/repos/{slug}/releases",
                            {"tag_name": tag, "name": title, "body": notes[:60000]})
        return str(made.get("html_url", ""))
