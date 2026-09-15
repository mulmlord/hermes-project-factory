"""Networked research: retrieving real sources, from inside the sandbox.

Research otherwise runs on whatever the local model happens to remember,
which is why the researcher is forced to label every answer's confidence.
This module is what turns a remembered claim into a citation: the researcher
names pages worth reading, the *sandbox* fetches them, the bytes are kept
under ``research/`` as evidence, and the extracted text goes back into the
prompt that answers the question.

Two properties matter more than the feature itself.

Fetching is off unless ``research.network`` is set, and even then it is
bounded: a page count per round, a byte cap per page, and a timeout.

Only public addresses are fetched. A container attached to the bridge
network can otherwise reach the operator's own inference server, a LAN
service, or a cloud metadata endpoint, and a model naming the URL is not a
reason to let it choose any of those targets. The check resolves the host
and rejects the *address*, because a public hostname is free to point at
127.0.0.1, and it re-runs on every redirect hop for the same reason.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
from dataclasses import dataclass, field, asdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from .log import Logger, NULL
from .workspace import slugify

#: Where fetched bytes are kept, relative to the project root.
STORE = "research/sources"

#: Runs in the sandbox, never in the Hermes process. Takes the URL, the
#: destination inside the workspace, the byte cap, the timeout, and whether
#: non-public addresses are permitted; prints the content type and the URL it
#: actually ended up at, which is the one worth citing.
_FETCH = r'''
import ipaddress, json, os, socket, sys
from urllib.parse import urlparse
from urllib.request import Request, HTTPRedirectHandler, build_opener

url, dest, limit, timeout, allow_private = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1")


def check(target):
    parts = urlparse(target)
    if parts.scheme not in ("http", "https"):
        raise ValueError("refused scheme %r" % parts.scheme)
    host = parts.hostname or ""
    if allow_private:
        return target
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port)
    except OSError as exc:
        raise ValueError("cannot resolve %s (%s)" % (host, exc))
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global or address.is_multicast:
            raise ValueError("refused non-public address %s -> %s" % (host, address))
    return target


class Guard(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check(newurl)
        return HTTPRedirectHandler.redirect_request(
            self, req, fp, code, msg, headers, newurl)


check(url)
opener = build_opener(Guard)
request = Request(url, headers={"User-Agent": "hermes-research (local factory)"})
with opener.open(request, timeout=timeout) as response:
    body = response.read(limit)
    meta = {"content_type": response.headers.get("Content-Type", ""),
            "final_url": response.geturl(), "status": response.status}
parent = os.path.dirname(dest)
if parent:
    os.makedirs(parent, exist_ok=True)
with open(dest, "wb") as handle:
    handle.write(body)
print(json.dumps(meta))
'''

_SUFFIXES = [("json", ".json"), ("text/plain", ".txt"), ("markdown", ".md"),
             ("xml", ".xml")]


@dataclass
class Source:
    """A page that was actually retrieved, and where its bytes were kept."""

    url: str
    path: str                 # workspace-relative copy of exactly what arrived
    title: str = ""
    bytes: int = 0
    fetched_at: float = field(default_factory=time.time)
    why: str = ""             # what the researcher wanted from it

    def cite(self) -> str:
        return f"{self.title or self.url} <{self.url}>"


class Librarian:
    """Fetches what the researcher asks for, within a fixed budget."""

    def __init__(self, sandbox, root: Path, cfg: dict, log: Logger = NULL):
        self.sandbox = sandbox
        self.root = Path(root)
        self.log = log
        self.max_pages = int(cfg.get("max_fetches", 6))
        self.max_bytes = int(cfg.get("fetch_bytes", 250_000))
        self.timeout = int(cfg.get("fetch_timeout", 30))
        #: Escape hatch for an internal documentation server, mirroring
        #: ``allow_remote`` on a model. Off by default, and deliberately not
        #: something a model's answer can turn on.
        self.allow_private = bool(cfg.get("allow_private", False))
        self.spent = 0
        self.fetched: dict[str, Source] = {}
        self.refusals: list[dict[str, str]] = []

    @property
    def remaining(self) -> int:
        return max(0, self.max_pages - self.spent)

    def fetch(self, url: str, why: str = "") -> Source | None:
        """Retrieve one page. Returns None when it was refused or failed."""
        url = (url or "").strip()
        if url in self.fetched:      # a round asking twice costs one fetch
            return self.fetched[url]
        problem = self._refuse(url)
        if problem:
            return self._refused(url, problem)

        destination = f"{STORE}/{self._filename(url)}"
        command = (
            f"python3 -c {shlex.quote(_FETCH)} {shlex.quote(url)} "
            f"{shlex.quote(destination)} {self.max_bytes} {self.timeout} "
            f"{'1' if self.allow_private else '0'}"
        )
        result = self.sandbox.run(command, timeout=self.timeout + 30, network=True)
        self.spent += 1
        if not result.ok:
            return self._refused(url, _reason(result))

        meta = _meta(result.stdout)
        stored = self.root / destination
        if not stored.is_file():
            return self._refused(url, "the sandbox reported success but wrote nothing")
        raw = stored.read_bytes()
        final = meta.get("final_url") or url
        suffix = _suffix(meta.get("content_type", ""))
        if suffix and stored.suffix != suffix:
            stored = stored.rename(stored.with_suffix(suffix))
            destination = str(stored.relative_to(self.root))
        source = Source(url=final, path=destination, title=title_of(raw),
                        bytes=len(raw), why=why)
        self.fetched[url] = source
        if final != url:
            self.fetched[final] = source
        self.log.info("research.fetched", url=final[:120], bytes=len(raw),
                      remaining=self.remaining)
        return source

    def gather(self, wanted: list) -> list[Source]:
        """Fetch a researcher's list of pages, stopping when the budget runs out."""
        found: list[Source] = []
        for item in wanted:
            if not self.remaining:
                self.log.info("research.budget_spent", pages=self.spent)
                break
            url = item.get("url") if isinstance(item, dict) else str(item)
            why = item.get("why", "") if isinstance(item, dict) else ""
            source = self.fetch(url or "", why)
            if source and source not in found:
                found.append(source)
        return found

    def excerpts(self, sources: list[Source], budget: int = 12000) -> str:
        """The retrieved text, labelled so the model can cite what it used."""
        if not sources:
            return ""
        share = max(1000, budget // len(sources))
        blocks = []
        for index, source in enumerate(sources, 1):
            text = self.text(source, share)
            if text:
                blocks.append(f"[{index}] {source.cite()}\n{text}")
        return "\n\n".join(blocks)

    def text(self, source: Source, limit: int) -> str:
        path = self.root / source.path
        if not path.is_file():
            return ""
        return to_text(path.read_bytes())[:limit]

    def _refuse(self, url: str) -> str:
        """Cheap local checks, so an obvious refusal costs no container."""
        if not url:
            return "empty url"
        parts = urlparse(url)
        if parts.scheme not in ("http", "https"):
            return f"refused scheme {parts.scheme or 'none'!r}"
        if not parts.hostname:
            return "no host in url"
        if not self.remaining:
            return f"fetch budget of {self.max_pages} page(s) is spent"
        return ""

    def _refused(self, url: str, reason: str) -> None:
        self.refusals.append({"url": url, "reason": reason})
        self.log.warn("research.fetch_refused", url=url[:120], reason=reason[:200])
        return None

    def _filename(self, url: str) -> str:
        parts = urlparse(url)
        stem = slugify(f"{parts.hostname or 'source'}-{Path(parts.path).name}", 60)
        # The digest keeps two pages with the same slug from overwriting each
        # other; research/ is evidence and must not lose a fetch silently.
        return f"{stem}-{hashlib.sha256(url.encode()).hexdigest()[:8]}.html"


class _Extract(HTMLParser):
    """Text and title out of HTML, using only the standard library."""

    SKIP = {"script", "style", "noscript", "svg", "head", "template"}
    BREAK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "section", "article", "pre", "table", "blockquote"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skipping = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skipping += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skipping:
            self._skipping -= 1
        if tag == "title":
            self._in_title = False
        if tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skipping:
            self.parts.append(data)


def to_text(raw: bytes | str) -> str:
    """Readable text from a fetched page, with markup and scripts removed."""
    markup = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    if not _looks_like_html(markup):
        return _collapse(markup)
    parser = _Extract()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # noqa: BLE001 - a malformed page still yields what parsed
        pass
    return _collapse("".join(parser.parts))


def title_of(raw: bytes | str) -> str:
    markup = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    if not _looks_like_html(markup):
        return ""
    parser = _Extract()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # noqa: BLE001
        pass
    return " ".join(parser.title.split())[:200]


def _looks_like_html(markup: str) -> bool:
    return "<" in markup[:4000] and re.search(r"<\s*[a-zA-Z!/]", markup[:4000]) is not None


def _collapse(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _meta(stdout: str) -> dict:
    for line in reversed((stdout or "").strip().splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _reason(result) -> str:
    """The sandbox's own words, which for a refusal is the ValueError text."""
    text = (result.stderr or result.stdout or "").strip()
    match = re.search(r"(?:ValueError|URLError|HTTPError|error):?\s*(.+)", text)
    detail = (match.group(1) if match else text.splitlines()[-1] if text else "")
    if result.timed_out:
        return f"timed out after {result.seconds:.0f}s"
    return detail[:300] or f"exit {result.exit_code}"


def _suffix(content_type: str) -> str:
    """A suffix that cannot make retrieved material look like project source.

    The store sits inside the project, and the security scan reads every file
    carrying a source suffix. A page fetched from ``.../setup.py`` must
    therefore not be saved as ``.py``: what was retrieved is evidence, never
    code, and must not be able to fail the project's own gate. So the suffix
    comes from a fixed list and never from the URL.
    """
    lowered = (content_type or "").lower()
    for marker, suffix in _SUFFIXES:
        if marker in lowered:
            return suffix
    return ".html"


def as_records(sources: list[Source]) -> list[dict]:
    return [asdict(s) for s in sources]
