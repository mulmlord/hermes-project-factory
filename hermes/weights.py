"""Downloading model weights from a Hugging Face repository, without its client.

``start.sh`` fetched weights with ``hf download``, the command-line client
that ``huggingface_hub`` installs. The function it served is small: list the
files of one revision of one repository, fetch each over HTTPS into a
directory, survive an interrupted multi-gigabyte transfer, and check that what
arrived is what the repository holds. This module does exactly that over the
Hub's public HTTP interface, from the standard library alone.

* Files are listed with ``GET /api/models/<repo>/tree/<revision>?recursive=true``,
  following the ``Link: rel="next"`` pages, and fetched from
  ``/<repo>/resolve/<revision>/<path>``.
* A partial download is kept as ``<file>.part`` and resumed with a ``Range``
  request, because a 5 GiB transfer that restarts from zero on every dropped
  connection may never finish.
* A file stored in LFS is checked against the SHA-256 the listing gives for
  it, and one that does not match is deleted rather than served.
* A path in the listing that would land outside the target directory is
  refused. The listing is a remote document, and an absolute path or a ``..``
  in it is a write anywhere on this machine.
* ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) authorises gated repositories,
  and is sent to the Hub only: resolve answers redirect to a storage host, and
  a token forwarded there is a token handed to whoever runs that host.
  ``HF_ENDPOINT`` points the whole thing at a mirror.

Run it as ``python -m hermes.weights REPO DIRECTORY``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ENDPOINT = "https://huggingface.co"
CHUNK = 1 << 20
REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class WeightsError(RuntimeError):
    pass


@dataclass
class RemoteFile:
    path: str
    size: int
    sha256: str = ""         # set for LFS files, which is every large one


def _token() -> str:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""


class _HubOnly(urllib.request.HTTPRedirectHandler):
    """Follow redirects, and drop the token when one leaves the Hub's host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and (urllib.parse.urlsplit(newurl).netloc
                                != urllib.parse.urlsplit(req.full_url).netloc):
            for name in list(new.headers):
                if name.lower() == "authorization":
                    del new.headers[name]
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_OPENER = urllib.request.build_opener(_HubOnly)


def _request(url: str, *, token: str = "", headers: dict | None = None,
             timeout: float = 60):
    request = urllib.request.Request(url, headers={"User-Agent": "hermes-weights/1",
                                                   **(headers or {})})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    return _OPENER.open(request, timeout=timeout)


def _next_page(link: str) -> str:
    for part in (link or "").split(","):
        match = re.match(r'\s*<([^>]+)>\s*;\s*rel="?next"?', part)
        if match:
            return match.group(1)
    return ""


def _safe(path: str) -> str:
    """*path* as a relative POSIX path inside the target, or a refusal."""
    pure = PurePosixPath(path)
    if (not path or pure.is_absolute() or ".." in pure.parts or "\\" in path
            or any(part in ("", ".") for part in path.split("/"))):
        raise WeightsError(f"refusing a path from the listing that leaves the "
                           f"target directory: {path!r}")
    return pure.as_posix()


def listing(repo: str, revision: str = "main", *, endpoint: str = ENDPOINT,
            token: str = "") -> list[RemoteFile]:
    """Every file in *repo* at *revision*."""
    if not REPO_ID.match(repo):
        raise WeightsError(f"not a repository id of the form owner/name: {repo!r}")
    url = (f"{endpoint.rstrip('/')}/api/models/{repo}/tree/"
           f"{urllib.parse.quote(revision, safe='')}?recursive=true")
    files: list[RemoteFile] = []
    while url:
        try:
            with _request(url, token=token) as response:
                page = json.loads(response.read().decode("utf-8"))
                url = _next_page(response.headers.get("Link", ""))
        except urllib.error.HTTPError as exc:
            hint = (" (a gated or private repository needs HF_TOKEN)"
                    if exc.code in (401, 403) else "")
            raise WeightsError(f"listing {repo}: HTTP {exc.code}{hint}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise WeightsError(f"listing {repo}: {exc}") from exc
        for entry in page if isinstance(page, list) else []:
            if not isinstance(entry, dict) or entry.get("type") != "file":
                continue
            lfs = entry.get("lfs") or {}
            files.append(RemoteFile(_safe(str(entry.get("path", ""))),
                                    int(lfs.get("size") or entry.get("size") or 0),
                                    str(lfs.get("oid") or "")))
    if not files:
        raise WeightsError(f"{repo}@{revision} lists no files")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


class _Progress:
    def __init__(self, stream, total: int):
        self.stream, self.total, self.done = stream, total, 0
        self.tty = hasattr(stream, "isatty") and stream.isatty()
        self.started = time.monotonic()
        self.shown = 0.0

    def advance(self, amount: int, label: str, final: bool = False) -> None:
        self.done += amount
        now = time.monotonic()
        if not self.tty or (not final and now - self.shown < 0.2):
            return
        self.shown = now
        rate = self.done / max(now - self.started, 1e-6)
        self.stream.write(f"\r    {_size(self.done)} of {_size(self.total)} "
                          f"({_size(rate)}/s)  {label[:50]}\033[K")
        self.stream.flush()

    def line(self, text: str) -> None:
        if self.tty:
            self.stream.write("\r\033[K")
        self.stream.write(f"    {text}\n")
        self.stream.flush()


def _size(count: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if count < 1024 or unit == "GiB":
            return f"{count:.1f} {unit}" if unit != "B" else f"{int(count)} B"
        count /= 1024
    return f"{count:.1f} GiB"


def fetch(repo: str, target: Path, file: RemoteFile, *, revision: str = "main",
          endpoint: str = ENDPOINT, token: str = "", progress: _Progress | None = None,
          attempts: int = 5) -> bool:
    """Download one file into *target*. Returns False when it was already there."""
    destination = Path(target) / file.path
    if destination.is_file() and destination.stat().st_size == file.size:
        if not file.sha256 or _sha256(destination) == file.sha256:
            if progress:
                progress.advance(file.size, file.path)
            return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    url = (f"{endpoint.rstrip('/')}/{repo}/resolve/"
           f"{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(file.path)}")
    for attempt in range(1, attempts + 1):
        have = partial.stat().st_size if partial.is_file() else 0
        if file.size and have > file.size:
            partial.unlink()
            have = 0
        if progress:
            progress.advance(have, file.path)
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with _request(url, token=token, headers=headers, timeout=120) as response:
                resumed = have and response.status == 206
                with partial.open("ab" if resumed else "wb") as out:
                    if have and not resumed and progress:
                        progress.advance(-have, file.path)   # the server restarted it
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        out.write(block)
                        if progress:
                            progress.advance(len(block), file.path)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and file.size and have == file.size:
                break                                  # already complete
            if exc.code in (401, 403, 404) or attempt == attempts:
                raise WeightsError(f"{file.path}: HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            if attempt == attempts:
                raise WeightsError(f"{file.path}: {exc}") from exc
        if progress:
            progress.advance(-(partial.stat().st_size if partial.is_file() else 0),
                             file.path)
        time.sleep(min(30, 2 ** attempt))
    received = partial.stat().st_size
    if file.size and received != file.size:
        raise WeightsError(f"{file.path}: received {received} bytes, expected {file.size}")
    if file.sha256 and _sha256(partial) != file.sha256:
        partial.unlink()
        raise WeightsError(f"{file.path}: SHA-256 does not match the repository; "
                           "the partial download was deleted")
    partial.replace(destination)
    return True


def download(repo: str, target: Path, *, revision: str = "main", endpoint: str = "",
             token: str | None = None, stream=None) -> list[str]:
    """Fetch every file of *repo* into *target*. Returns the paths downloaded."""
    endpoint = endpoint or os.environ.get("HF_ENDPOINT") or ENDPOINT
    token = _token() if token is None else token
    stream = stream or sys.stderr
    files = listing(repo, revision, endpoint=endpoint, token=token)
    progress = _Progress(stream, sum(f.size for f in files))
    progress.line(f"{len(files)} files, {_size(progress.total)} in {repo}@{revision}")
    fetched = []
    for file in files:
        if fetch(repo, Path(target), file, revision=revision, endpoint=endpoint,
                 token=token, progress=progress):
            fetched.append(file.path)
            progress.line(f"{file.path} ({_size(file.size)})")
    progress.line(f"done: {len(fetched)} downloaded, {len(files) - len(fetched)} "
                  "already present")
    return fetched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hermes.weights",
        description="Download a Hugging Face model repository into a directory.")
    parser.add_argument("repo", help="owner/name")
    parser.add_argument("directory")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="", help="a mirror (default HF_ENDPOINT "
                                                       "or huggingface.co)")
    args = parser.parse_args(argv)
    try:
        download(args.repo, Path(args.directory), revision=args.revision,
                 endpoint=args.endpoint)
    except WeightsError as exc:
        print(f"weights: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nweights: interrupted; run the same command again to resume",
              file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
