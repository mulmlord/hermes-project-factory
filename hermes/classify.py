"""A local text classifier for the one decision that is a label, not prose.

:func:`hermes.verify.scan_unsafe` is eight regular expressions. They catch
``os.system`` with a ``+`` in it and miss ``os.popen``, ``yaml.load`` without a
safe loader, ``pickle.loads`` on a socket, zip-slip, and every path traversal
ever written. Generated code is untrusted here, and a pattern list is the
thinnest possible reading of that.

A classifier closes part of the gap for the price of a forward pass over a few
hundred short chunks. What it cannot do is be believed on its own: an un-tuned
checkpoint scores near chance, and a false positive that fails a correct ticket
costs a whole generate/verify/triage cycle -- the same trap
:func:`hermes.verify.review` already avoids by refusing to guess at defects.

So a finding from here is advisory. It is written into the security gate's
detail, which the reviewer and the triager already read, and the expensive
model decides what to do about it. The regexes keep their veto; this only ever
adds text to a report, unless an operator who has tuned a checkpoint sets
``security.classifier.block``. Cheap model flags, expensive model adjudicates.

The service is spoken to over HTTP for the same reason the inference backends
are: a classifier wants torch and Hermes does not. ``contrib/laya_server.py``
is a reference adapter over the ``laya`` SDK. It answers::

    POST /classify
    {"question": "...", "options": ["safe", ...],
     "items": [{"id": "c0", "text": "<source chunk>"}]}

    -> {"results": [{"id": "c0", "label": "safe",
                     "confidence": 0.97, "escalate": false}]}

    GET /health -> 200

When nothing answers, the gate is exactly what it was before.

The same service is also asked a second question it is not yet trusted to
answer: what class of failure a broken attempt is. The triager -- a full
reasoning-model call per failure -- answers that today, and the answer is one
word out of six, which is a classifier's shape of decision. Whether a cheap
model could take it over is an empirical question, so it is asked in *shadow*:
the guess is written to :mod:`hermes.ledger` beside the triager's answer and
never read by the loop. ``hermes ledger --shadow`` reports how often the two
agree, which is the evidence any later promotion has to be argued from.
A request may carry ``criteria``, a label -> description mapping, for
questions the adapter has no descriptions of its own for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Config, ConfigError, is_local_url
from .log import Logger, NULL
from .models import ModelError, _get, _post

#: The question the security scan asks, phrased so that "safe" is a real answer
#: rather than the absence of one. A classifier given only failure labels has
#: nowhere to put correct code.
SECURITY_QUESTION = (
    "Does this code contain a security vulnerability an attacker could exploit?"
)

#: The answer space. A scorer splits a fixed head budget across the option
#: markers, so these names stay short: a taxonomy written as sentences spends
#: its tokens on grammar instead of meaning. Kept well clear of the
#: high-cardinality regime where per-label budget collapses.
SECURITY_LABELS: tuple[str, ...] = (
    "safe",
    "command injection",
    "unsafe deserialization",
    "path traversal",
    "hardcoded credential",
    "unsafe eval",
    "server-side request forgery",
    "sql injection",
    "weak cryptography",
)

#: The shadow question, asked of the tail of a failed attempt's output.
TRIAGE_QUESTION = "What is the root cause of this build or test failure?"

#: The triager's own vocabulary and its own definitions of it (see
#: ``hermes.prompts.TRIAGE``), so that agreement between the two is agreement
#: about the same six words and not a translation between taxonomies.
TRIAGE_LABELS: dict[str, str] = {
    "trivial": "a typo, a missing import, a wrong path",
    "implementation": "the code is wrong",
    "architecture": "the design cannot work",
    "research_gap": "we do not know enough to fix it",
    "environment": "a missing tool or dependency",
    "flaky": "the failure does not depend on the code; it passes on a rerun",
}

#: Characters one chunk shows the classifier. The smallest useful checkpoints
#: take 512 tokens, so this stays near a quarter of that in characters and
#: leaves the question and the option markers room to fit beside it.
CHUNK_CHARS = 1200

#: Lines repeated between neighbouring chunks, so a call split across a window
#: boundary is whole in at least one of them.
CHUNK_OVERLAP = 3

#: Chunks sent in one request. Batching is the whole performance argument for
#: a classifier, but a single payload carrying a thousand chunks is a timeout.
BATCH = 64


def chunks(text: str, *, size: int = CHUNK_CHARS,
           overlap: int = CHUNK_OVERLAP) -> list[tuple[int, str]]:
    """Split source into overlapping windows, as ``(first line number, text)``.

    Windows are cut on line boundaries, not character counts: a finding has to
    name a line, and a chunk starting mid-statement reads as noise to whatever
    scores it. A single line longer than *size* still gets its own window --
    minified sources must not stall the walk.
    """
    lines = text.splitlines()
    out: list[tuple[int, str]] = []
    start = 0
    while start < len(lines):
        used, end = 0, start
        while end < len(lines) and (end == start or used + len(lines[end]) + 1 <= size):
            used += len(lines[end]) + 1
            end += 1
        out.append((start + 1, "\n".join(lines[start:end])))
        if end >= len(lines):
            break
        start = max(end - overlap, start + 1)
    return out


@dataclass(frozen=True)
class Finding:
    """One possible defect, in the form the gate report shows it."""

    path: str
    line: int
    label: str
    confidence: float

    def render(self) -> str:
        return (f"{self.path}:{self.line}: possible {self.label} "
                f"({self.confidence:.0%} confidence, unconfirmed)")


@dataclass(frozen=True)
class Guess:
    """One label the classifier chose, and how sure it said it was."""

    label: str
    confidence: float


@dataclass
class ClassifierConfig:
    enabled: bool = False
    base_url: str = "http://localhost:6768"
    model: str = "laya"
    timeout: int = 30
    #: Below this the answer is noise. Calibration is what makes a threshold
    #: mean anything, so a checkpoint that ships over-confident must have its
    #: temperature refitted before this number is worth setting.
    threshold: float = 0.75
    #: Let a finding fail the gate. Off until an operator has tuned a
    #: checkpoint on their own code and measured its false positive rate.
    block: bool = False
    #: Chunks one scan may classify, across every file it was given.
    max_chunks: int = 400
    #: Also guess each failure's triage class, for the ledger only. Costs one
    #: request per triaged failure and changes no decision the loop makes.
    shadow: bool = True
    allow_remote: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class Classifier:
    """A classification service, or nothing at all.

    Every failure mode -- disabled, unreachable, malformed answer, service
    error -- ends the same way: no findings, and a gate that behaves as it did
    before anyone configured this. A security scan that breaks the build when
    its helper is down is worse than no security scan.
    """

    def __init__(self, cfg: ClassifierConfig, log: Logger = NULL):
        self.cfg = cfg
        self.log = log
        self._up: bool | None = None

    @classmethod
    def from_config(cls, config: Config, log: Logger = NULL) -> "Classifier | None":
        """Build one from ``security.classifier``, or ``None`` when disabled."""
        spec = dict(config.get("security.classifier") or {})
        if not spec.pop("enabled", False):
            return None
        known = {f for f in ClassifierConfig.__dataclass_fields__
                 if f not in ("extra", "enabled")}
        # enabled is passed explicitly because it was popped off above: taking
        # it from the remaining keys would leave the flag at its default and
        # build a classifier that answers nothing.
        cfg = ClassifierConfig(enabled=True,
                               **{k: v for k, v in spec.items() if k in known},
                               extra={k: v for k, v in spec.items() if k not in known})
        if not cfg.allow_remote and not is_local_url(cfg.base_url):
            raise ConfigError(
                f"security.classifier: {cfg.base_url} is not a local address. "
                "Hermes is local-model-only; set allow_remote to override."
            )
        return cls(cfg, log=log)

    def available(self) -> bool:
        """Whether the service answered, probed once per classifier."""
        if self._up is None:
            try:
                _get(f"{self.cfg.base_url.rstrip('/')}/health", timeout=5)
                self._up = True
            except ModelError as exc:
                self.log.info("classify.absent", url=self.cfg.base_url, error=str(exc)[:120])
                self._up = False
        return self._up

    def ask(self, question: str, options: list[str],
            items: list[dict], criteria: dict[str, str] | None = None) -> list[dict]:
        """Classify *items*, batched. Returns whatever came back, or ``[]``."""
        url = f"{self.cfg.base_url.rstrip('/')}/classify"
        out: list[dict] = []
        for at in range(0, len(items), BATCH):
            payload = {"model": self.cfg.model, "question": question,
                       "options": options, "items": items[at:at + BATCH],
                       **({"criteria": criteria} if criteria else {}),
                       **self.cfg.extra}
            try:
                answer = _post(url, payload, self.cfg.timeout)
            except ModelError as exc:
                # One bad batch must not lose the batches that worked.
                self.log.warn("classify.failed", url=url, error=str(exc)[:200])
                self._up = False
                break
            results = answer.get("results")
            if isinstance(results, list):
                out.extend(r for r in results if isinstance(r, dict))
        return out

    def guess(self, question: str, criteria: dict[str, str],
              text: str) -> Guess | None:
        """Pick one of *criteria*'s labels for *text*, or ``None``.

        ``None`` covers every way of not having an answer -- disabled, down,
        or a label outside the options -- because a shadow guess that is
        wrong about its own vocabulary is not a guess anyone can score.
        """
        if not self.cfg.enabled or not text.strip() or not self.available():
            return None
        for result in self.ask(question, list(criteria),
                               [{"id": "g0", "text": text}], criteria):
            label = str(result.get("label") or "").strip()
            try:
                confidence = float(result.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if label in criteria:
                return Guess(label, confidence)
        return None

    def scan(self, sources: dict[str, str]) -> list[Finding]:
        """Classify every chunk of *sources*, keeping what looks like a defect.

        *sources* maps a display path to its text. Overlapping windows mean one
        defect can be flagged twice from different start lines, so a path keeps
        only its most confident finding per label: the gate detail is read by a
        model with a context budget, and the same warning three times spends it
        without saying anything new.
        """
        if not self.cfg.enabled or not sources or not self.available():
            return []
        items: list[dict] = []
        index: dict[str, tuple[str, int]] = {}
        for path, text in sources.items():
            for line, chunk in chunks(text):
                key = f"c{len(items)}"
                index[key] = (str(path), line)
                items.append({"id": key, "text": chunk})
        if len(items) > self.cfg.max_chunks:
            self.log.info("classify.truncated", chunks=len(items),
                          limit=self.cfg.max_chunks)
            items = items[: self.cfg.max_chunks]

        best: dict[tuple[str, str], Finding] = {}
        for result in self.ask(SECURITY_QUESTION, list(SECURITY_LABELS), items):
            where = index.get(str(result.get("id")))
            label = str(result.get("label") or "").strip()
            if where is None or not label or label == "safe":
                continue
            try:
                confidence = float(result.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            # An escalation is the service saying it does not know, which is
            # precisely what an advisory is for: it goes to the reviewer even
            # though it did not clear the threshold on its own.
            if confidence < self.cfg.threshold and not result.get("escalate"):
                continue
            path, line = where
            finding = Finding(path, line, label, confidence)
            key = (path, label)
            if key not in best or confidence > best[key].confidence:
                best[key] = finding
        findings = sorted(best.values(), key=lambda f: (-f.confidence, f.path, f.line))
        if findings:
            self.log.info("classify.findings", count=len(findings),
                          files=len({f.path for f in findings}))
        return findings
