"""What ``hermes doctor --probe`` asks a live server, and which layer broke.

"The local model cannot use its tools" is most often not the model. It is the
server: a schema it accepted and then ignored, a context it was started too
small for, weights that spilled onto the CPU when the context grew. Each of
those looks, from inside a run, exactly like a model that is bad at the task,
and each costs a generation per ticket to rediscover. So the probe asks one
small question per layer, and says which layer answered wrongly.

Not the server's own tool-call parser. Hermes never sends ``tools=``: its
lookups travel as JSON in the reply body (see :mod:`hermes.tools`), so a
server whose chat template mangles ``<tool_call>`` blocks costs it nothing,
and a probe of that parser would fail servers Hermes runs on perfectly well.
What it does depend on is asked instead:

* **context** -- the window the server was started with, against the one the
  config sizes every prompt to. A config larger than the server is the usual
  reason a clean install fails on the first long prompt.
* **schema** -- a prompt asking for one shape under a schema demanding
  another. A server that enforces the schema answers in the schema's shape; a
  server that accepts the parameter and ignores it follows the prompt, which
  is the one outcome no refusal would have revealed.
* **lookup** -- the real lookup protocol and a question only a file read can
  answer. A model that guesses rather than asks is the omission failure; one
  that asks in a shape the toolbox cannot run is the malformed one.
* **memory** -- where the server re-places the model per request (ollama),
  how much of it ended up outside VRAM. A spill degrades the reply's format,
  not only its speed.

Each probe is one sample at temperature 0. A warning is a tendency worth
knowing about, not a verdict on the model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .models import Backend, ModelError, _rejects_parameter, extract_json, strip_reasoning
from .schema import NUMBER, OFF, SCHEMA, any_of, check, enum, obj
from .tools import LOOKUP_REQUEST, Toolbox, requested

OK, WARN, FAIL = "ok", "warn", "fail"

#: A value no model writes unless something made it: the prompt never
#: mentions it, so a reply carrying it came from the decoder's grammar.
MARK = "hermes-schema-held"
SHAPE = obj({"probe": enum(MARK)}, required=("probe",))
SHAPE_PROMPT = 'Reply with exactly this JSON object and nothing else: {"status": "ready"}'

ANSWER = obj({"timeout": NUMBER}, required=("timeout",))
LOOKUP_PROMPT = (
    "TASK: the project's src/settings.py defines a constant TIMEOUT. Reply "
    'with {"timeout": <its value>}. You have not been shown that file.\n\n')

#: The argument each tool cannot run without. Anything else the toolbox
#: defaults, and a request missing one of these is refused with a reason.
NEEDS = {"read_file": "path", "grep": "pattern", "outline": "path",
         "find_symbol": "name", "find_references": "name", "recall": "query"}

SYSTEM = "You are a software engineer on an automated team. Reply with JSON only."
#: Room for a reasoning model's <think> block before the JSON it is asked for.
TOKENS = 512


@dataclass(frozen=True)
class Finding:
    layer: str
    verdict: str
    text: str


def probe(backend: Backend) -> list[Finding]:
    """Every layer of *backend*, in the order a run would meet them.

    The schema probe goes first because it is the one that may walk the
    backend's level down, and the lookup probe should be asked the way a run
    would then ask it. Memory goes last: ollama lists a model only once
    something has loaded it.
    """
    findings = [context(backend), *shape(backend)]
    if not any(f.verdict == FAIL for f in findings[1:]):
        findings.append(lookup(backend))
    placed = memory(backend)
    return findings + ([placed] if placed else [])


def context(backend: Backend) -> Finding:
    cfg = backend.cfg
    served = backend.served_context()
    if served is None:
        return Finding("context", OK, f"{cfg.context} configured; this server does "
                       "not report its own, so that is unchecked")
    if cfg.context > served:
        return Finding("context", FAIL, (
            f"models.{cfg.role}.context is {cfg.context} but the server serves "
            f"{served}: every prompt sized past {served} tokens is refused. Set "
            f"models.{cfg.role}.context: {served}, or restart the server larger"))
    return Finding("context", OK, f"{cfg.context} of the {served} the server serves")


def shape(backend: Backend) -> list[Finding]:
    """Whether the server holds a reply to a schema, and at which level it can."""
    found: list[Finding] = []
    while True:
        level = backend.effective(SHAPE)
        try:
            raw = backend.complete(SYSTEM, SHAPE_PROMPT, temperature=0,
                                   max_tokens=TOKENS, json_mode=level != OFF,
                                   schema=SHAPE if level == SCHEMA else None)
            break
        except ModelError as exc:
            if level == OFF or not _rejects_parameter(exc):
                return found + [Finding("schema", FAIL, f"the server failed the "
                                        f"request: {str(exc)[:200]}")]
            # Exactly what a run does on the same refusal, so what is
            # reported below is the level a run would actually get.
            found.append(Finding("schema", WARN, (
                f"refused structured output at level {level!r}; runs walk down "
                f"to {backend.weaken(SHAPE)!r} on the first call. Setting "
                f"models.{backend.cfg.role}.structured_output to it saves the "
                "refused round trip")))
    try:
        value = extract_json(raw)
    except ValueError:
        return found + [Finding("schema", FAIL, (
            f"no JSON in the reply at level {level!r}: every phase will spend its "
            f"repair attempts on this. The reply began "
            f"{strip_reasoning(raw)[:120]!r}"))]
    if level != SCHEMA:
        return found + [Finding("schema", OK, (
            f"level {level!r}: replies parse, and their shape is checked in Python, "
            "one retry per miss"))]
    if not check(SHAPE, value):
        return found + [Finding("schema", OK, "enforced: the reply took the schema's "
                                "shape over the prompt's")]
    return found + [Finding("schema", WARN, (
        f"accepted json_schema and ignored it (replied {_brief(value)}): a shape "
        "is held only by the Python check, one retry per miss. A vLLM without "
        "a guided-decoding backend, or a proxy that drops response_format, "
        "does this"))]


def lookup(backend: Backend) -> Finding:
    """Whether the model asks for a file it needs, in a form that would run."""
    toolbox = Toolbox(None)
    schema = any_of(ANSWER, LOOKUP_REQUEST)
    level = backend.effective(schema)
    try:
        raw = backend.complete(SYSTEM, LOOKUP_PROMPT + toolbox.protocol(),
                               temperature=0, max_tokens=TOKENS,
                               json_mode=level != OFF,
                               schema=schema if level == SCHEMA else None)
        value = extract_json(raw)
    except ModelError as exc:
        return Finding("lookup", FAIL, f"the server failed the request: {str(exc)[:200]}")
    except ValueError:
        return Finding("lookup", FAIL, "no JSON in the reply to the lookup protocol")
    requests = requested(value)
    if requests:
        bad = [r for r in requests if not _runnable(r, toolbox.available)]
        if not bad:
            return Finding("lookup", OK, f"asked before answering: {_brief(requests[0])}")
        return Finding("lookup", WARN, (
            f"asked to look, in a shape the toolbox refuses: {_brief(bad[0])}. "
            "Each such request spends a lookup on an explanation"))
    if isinstance(value, dict) and value.get("timeout") is not None:
        return Finding("lookup", WARN, (
            f"answered without looking, guessing timeout={value['timeout']!r} from "
            "a file it was never shown. Expect agents to write against names "
            "they have not checked"))
    return Finding("lookup", WARN, f"replied with neither an answer nor a lookup: "
                   f"{_brief(value)}")


def memory(backend: Backend) -> Finding | None:
    placed = backend.placement()
    if not placed or not placed[0]:
        return None
    size, vram = placed
    if vram >= size:
        return Finding("memory", OK, f"all {size >> 20} MiB in VRAM")
    return Finding("memory", WARN, (
        f"{(size - vram) * 100 // size}% of the model is in system RAM at a "
        f"{backend.cfg.context}-token context: format reliability drops with it, "
        f"not only speed. Lower models.{backend.cfg.role}.context or use a "
        "smaller quantisation"))


def _runnable(request, available) -> bool:
    if not isinstance(request, dict):
        return False
    tool = str(request.get("tool") or "")
    need = NEEDS.get(tool)
    return tool in available and (need is None or bool(request.get(need)))


def _brief(value) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 100 else text[:97] + "..."
