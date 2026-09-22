"""Local inference backends and the task -> model router.

Only locally hosted servers are supported: Ollama's native API and any
OpenAI-compatible server (llama.cpp ``server``, vLLM, text-generation-webui).
There is deliberately no cloud fallback anywhere in this module.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable

from .config import Config, ConfigError, ModelConfig
from .log import Logger, NULL
from .schema import JSON, LEVELS, OFF, SCHEMA, check

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class ModelError(RuntimeError):
    pass


class PromptTooLarge(ModelError):
    """A prompt refused here, before a server had to refuse it.

    Raised locally so the failure names the budget and the blocks that blew
    it, rather than arriving as an HTTP 400 after a round trip.
    """


def _post(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:  # surface the server's own message
        raise ModelError(f"{url}: HTTP {exc.code} {exc.read()[:400].decode(errors='replace')}")
    except urllib.error.URLError as exc:
        raise ModelError(f"{url}: cannot reach local inference server ({exc.reason})")


def _get(url: str, timeout: int = 10) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001 - availability probe, any failure is "absent"
        raise ModelError(f"{url}: {exc}")


class Backend:
    """Common interface: turn a system+user prompt into text."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        #: How hard this server is currently asked to hold a reply to its
        #: shape, from :data:`hermes.schema.LEVELS`. It starts at whatever the
        #: config asks for and only ever walks down, when a server says it
        #: does not speak the parameter.
        self.level = cfg.structured_output

    def effective(self, schema: dict | None) -> str:
        """The strongest rung this particular call can actually use.

        A caller with no schema cannot ask for one, so it is asking for
        ``json`` however the config is set. Naming that here keeps
        :meth:`weaken` honest: the rung that was refused is the rung that
        gets dropped, and a refusal never costs a retry identical to the
        request that caused it.
        """
        return JSON if self.level == SCHEMA and not schema else self.level

    def weaken(self, schema: dict | None = None) -> str:
        """Drop below the rung this call actually used.

        One rung at a time, because the rungs are separate features: vLLM
        builds older than XGrammar take ``json_object`` and refuse
        ``json_schema``, and throwing away both over one refusal would cost
        every later call the guarantee the server does support.
        """
        self.level = LEVELS[max(0, LEVELS.index(self.effective(schema)) - 1)]
        return self.level

    def complete(self, system: str, prompt: str, *, temperature: float | None = None,
                 max_tokens: int | None = None, json_mode: bool = False,
                 schema: dict | None = None) -> str:
        raise NotImplementedError

    def available_models(self) -> list[str]:
        raise NotImplementedError

    def served_context(self) -> int | None:
        """The context window the server was started with, where it says.

        None means the server does not report one, not that it has none.
        """
        return None

    def placement(self) -> tuple[int, int] | None:
        """Bytes of the loaded model in all, and of those in VRAM.

        None where the question does not arise. vLLM and llama.cpp place the
        weights once at startup and fail loudly when they do not fit; only a
        server that re-places the model per request can quietly spill it.
        """
        return None


class OllamaBackend(Backend):
    def complete(self, system, prompt, *, temperature=None, max_tokens=None,
                 json_mode=False, schema=None):
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            # Verification runs for minutes between calls, which is long enough
            # for ollama to evict the model and charge a full reload next time.
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "num_ctx": self.cfg.context,
                **({"num_predict": max_tokens} if max_tokens else {}),
                **self.cfg.extra.get("options", {}),
            },
        }
        # ollama takes the schema document itself in the same field that takes
        # the string "json", and has since 0.5.
        level = self.effective(schema)
        if json_mode and level == SCHEMA:
            payload["format"] = schema
        elif json_mode and level != OFF:
            payload["format"] = "json"
        data = _post(f"{self.cfg.base_url.rstrip('/')}/api/chat", payload, self.cfg.timeout)
        return (data.get("message") or {}).get("content", "")

    def available_models(self):
        data = _get(f"{self.cfg.base_url.rstrip('/')}/api/tags")
        return [m["name"] for m in data.get("models", [])]

    def placement(self):
        # ollama sizes the KV cache from the num_ctx each request sends, and a
        # context that no longer fits beside the weights pushes layers onto
        # the CPU without a word. /api/ps is the only place that shows, and
        # it lists a model only while it is loaded.
        try:
            data = _get(f"{self.cfg.base_url.rstrip('/')}/api/ps")
        except ModelError:
            return None
        for entry in data.get("models", []):
            names = [str(entry.get("name", "")), str(entry.get("model", ""))]
            if _model_present(self.cfg.model, names):
                return _int(entry.get("size")), _int(entry.get("size_vram"))
        return None


class OpenAICompatBackend(Backend):
    """llama.cpp server, vLLM, and anything else speaking /v1/chat/completions."""

    def _base(self) -> str:
        base = self.cfg.base_url.rstrip("/")
        return base if base.endswith("/v1") else base + "/v1"

    def complete(self, system, prompt, *, temperature=None, max_tokens=None,
                 json_mode=False, schema=None):
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        level = self.effective(schema)
        if json_mode and level == SCHEMA:
            # vLLM compiles this to a grammar and masks the tokens that would
            # leave the shape; llama.cpp's server does the same through its
            # own GBNF conversion. A server that does neither answers 400
            # naming the parameter, which is what weaken() is for.
            # No "strict": that flag means OpenAI's dialect, which demands
            # every property be required and additional ones forbidden --
            # the opposite of the forgiving shapes hermes.schema.obj builds.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "hermes_reply", "schema": schema},
            }
        elif json_mode and level != OFF:
            payload["response_format"] = {"type": "json_object"}
        data = _post(f"{self._base()}/chat/completions", payload, self.cfg.timeout)
        choices = data.get("choices") or []
        if not choices:
            raise ModelError(f"empty completion from {self.cfg.base_url}")
        return choices[0].get("message", {}).get("content", "")

    def available_models(self):
        data = _get(f"{self._base()}/models")
        return [m["id"] for m in data.get("data", [])]

    def served_context(self):
        # vLLM puts --max-model-len on the model listing; llama.cpp's server
        # keeps n_ctx on /props, beside /v1 rather than under it.
        try:
            for entry in _get(f"{self._base()}/models").get("data", []):
                if (_model_present(self.cfg.model, [str(entry.get("id", ""))])
                        and _int(entry.get("max_model_len"))):
                    return _int(entry["max_model_len"])
        except ModelError:
            return None
        try:
            props = _get(f"{self._base()[:-len('/v1')]}/props")
        except ModelError:
            return None
        settings = props.get("default_generation_settings") or {}
        return _int(settings.get("n_ctx") or props.get("n_ctx")) or None


BACKENDS = {
    "ollama": OllamaBackend,
    "openai": OpenAICompatBackend,
    "llamacpp": OpenAICompatBackend,
    "vllm": OpenAICompatBackend,
}


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks emitted by reasoning models such as qwen3."""
    return _THINK.sub("", text or "").strip()


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of a model response.

    Local models wrap JSON in prose, fences and reasoning traces. Try, in
    order: the whole string, fenced blocks, then a brace/bracket scan.
    """
    cleaned = strip_reasoning(text)
    for candidate in _json_candidates(cleaned):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no JSON found in model output: {cleaned[:300]!r}")


def _json_candidates(text: str) -> Iterable[str]:
    text = text.strip()
    yield text
    for block in _FENCE.findall(text):
        yield block.strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            end = _match(text, start, opener, closer)
            if end != -1:
                yield text[start : end + 1]
            start = text.find(opener, start + 1)


def _match(text: str, start: int, opener: str, closer: str) -> int:
    """Index of the bracket closing the one at *start*, honouring strings."""
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i
    return -1


#: Characters per token assumed by every budget in the factory. Real
#: tokenizers give roughly 3.5-4 for prose and a little under that for dense
#: code; assuming 3 over-counts on purpose, because the two ways of being
#: wrong cost very different amounts. A prompt refused here costs a log line
#: and a smaller prompt. A prompt the server refuses costs a round trip, a
#: generation, and -- before this existed -- a whole phase. Hermes ships no
#: tokenizer: one per model family would be a dependency for a number that
#: only has to be right to a few percent.
CHARS_PER_TOKEN = 3

#: Tokens the chat template, role markers and BOS add on top of the text.
TEMPLATE_OVERHEAD = 200

#: What each task kind keeps back for its answer: a fraction of the configured
#: context, with a floor and a ceiling. This describes the shape of the answer
#: the role is asked for, not the model: the implementer returns whole files,
#: the reviewer returns a verdict and a few sentences.
RESERVE = {
    "coding": (0.35, 1024, 8192),
    "evolution": (0.35, 1024, 8192),
    "architecture": (0.30, 1024, 6144),
    "research": (0.30, 1024, 6144),
    "review": (0.20, 512, 3072),
    "simple": (0.20, 512, 2048),
}
DEFAULT_RESERVE = (0.25, 512, 4096)


def estimate_tokens(text: str) -> int:
    """Pessimistic token count for *text*. See :data:`CHARS_PER_TOKEN`."""
    return -(-len(text) // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Allowance:
    """How many tokens one task kind may spend, and on what.

    Produced only by :meth:`Router.allowance`, which is the single place
    ``models.<role>.context`` becomes a number the rest of the factory is
    allowed to use. Raising a model's configured context to match the server
    it actually runs on is therefore the only thing an operator has to do to
    give every agent more room.
    """

    task: str
    model: str
    context: int
    reserve: int
    overhead: int = TEMPLATE_OVERHEAD

    @property
    def prompt_tokens(self) -> int:
        """What the prompt may spend once the answer has its room."""
        return max(0, self.context - self.reserve - self.overhead)

    @property
    def prompt_chars(self) -> int:
        return self.prompt_tokens * CHARS_PER_TOKEN

    def check(self, *parts: str) -> int:
        """Token cost of *parts* together, or raise if they do not fit."""
        used = sum(estimate_tokens(part) for part in parts)
        if used > self.prompt_tokens:
            raise PromptTooLarge(
                f"task {self.task!r}: prompt is about {used} tokens but only "
                f"{self.prompt_tokens} fit -- {self.model} is configured with a "
                f"{self.context}-token context, of which {self.reserve} is held "
                f"back for the answer and {self.overhead} for the chat template. "
                "Shorten the prompt, or raise models.<role>.context if the "
                "server's --max-model-len is larger than the config says."
            )
        return used


class Router:
    """Resolves a task kind to a backend, with a small usage ledger."""

    def __init__(self, config: Config, log: Logger = NULL, governor=None):
        self.config = config
        self.log = log
        self._cache: dict[str, Backend] = {}
        self.usage: list[dict[str, Any]] = []
        #: Optional :class:`~hermes.resources.Governor`. Every completion is
        #: registered with it, which is what lets it know whether the GPU is
        #: already busy before it admits a second concurrent generation.
        self.governor = governor

    def backend(self, task: str) -> Backend:
        if task not in self._cache:
            cfg = self.config.model_for(task)
            self._cache[task] = BACKENDS[cfg.backend](cfg)
        return self._cache[task]

    def describe(self, task: str) -> str:
        cfg = self.config.model_for(task)
        return f"{cfg.backend}:{cfg.model}"

    def allowance(self, task: str) -> Allowance:
        """The token budget for *task*, derived from its configured context."""
        cfg = self.backend(task).cfg
        fraction, floor, ceiling = RESERVE.get(task, DEFAULT_RESERVE)
        reserve = min(max(int(cfg.context * fraction), floor), ceiling)
        # The floor is what a useful answer needs, but on a small context it
        # would leave the prompt nothing; half the window is the hard cap.
        return Allowance(task, cfg.model, cfg.context,
                         min(reserve, max(1, cfg.context // 2)))

    def complete(self, task: str, system: str, prompt: str, **kw: Any) -> str:
        backend = self.backend(task)
        allowance = self.allowance(task)
        used = allowance.check(system, prompt)
        # Tell the server how much room the answer gets. Left unsaid, a server
        # whose default max_tokens is small truncates every answer into
        # unparseable JSON, and one with no default lets a looping generation
        # run to the request timeout -- minutes of GPU for nothing.
        kw.setdefault("max_tokens", allowance.reserve)
        started = time.time()
        with self.governor.inference() if self.governor else nullcontext():
            text = backend.complete(system, prompt, **kw)
        elapsed = time.time() - started
        self.usage.append(
            {"task": task, "model": backend.cfg.model, "seconds": round(elapsed, 2),
             "chars_in": len(system) + len(prompt), "chars_out": len(text),
             "tokens_in": used}
        )
        self.log.debug("model.complete", task=task, model=backend.cfg.model,
                       seconds=round(elapsed, 1), chars_out=len(text))
        return strip_reasoning(text)

    def _json_attempt(self, task: str, system: str, prompt: str, kw: dict,
                      schema: dict | None = None) -> str:
        """One completion asked for under constrained decoding where supported.

        Walks down :data:`hermes.schema.LEVELS` as the server refuses each
        rung, so an old or minimal local server ends up answering unconstrained
        rather than failing the task. The walk is remembered on the backend:
        the probe costs one round trip per run, not one per call.
        """
        backend = self.backend(task)
        while True:
            try:
                return self.complete(task, system, prompt, json_mode=True,
                                     schema=schema, **kw)
            except ModelError as exc:
                if backend.effective(schema) == OFF or not _rejects_parameter(exc):
                    raise
                # The server does not speak this parameter. Losing the
                # guarantee is survivable; failing the task over a capability
                # probe is not.
                was = backend.effective(schema)
                self.log.warn("model.structured_output_downgraded", task=task,
                              was=was, now=backend.weaken(schema),
                              error=str(exc)[:200])

    def complete_json(self, task: str, system: str, prompt: str, *, attempts: int = 3,
                      schema: dict | None = None, **kw: Any) -> Any:
        """Completion that must parse as JSON, and should match *schema*.

        Constrained decoding makes a malformed or misshapen reply structurally
        impossible on servers that support it, so the repair loop is the
        fallback for the ones that do not rather than the normal path. Each
        repair costs a whole generation, which on a local model is the most
        expensive thing here.

        A reply that parses but departs from the schema is retried while
        attempts remain and then returned anyway. Every caller already reads
        its reply with ``.get`` and a default, so a missing key is a thinner
        answer; refusing it outright would turn a thin answer into a failed
        phase, which is a worse trade than the one the schema is here to make.
        """
        last_error = ""
        current = prompt
        for attempt in range(1, attempts + 1):
            text = self._json_attempt(
                task, system + "\n\nRespond with JSON only. No prose.", current, kw,
                schema)
            try:
                value = extract_json(text)
            except ValueError as exc:
                last_error = str(exc)
                self.log.warn("model.json_retry", task=task, attempt=attempt)
                current = (
                    f"{prompt}\n\nYour previous reply was not valid JSON ({exc}). "
                    "Reply with a single valid JSON value and nothing else."
                )
                kw["temperature"] = min(0.8, kw.get("temperature", 0.2) + 0.2 * attempt)
                continue
            faults = check(schema, value) if schema else []
            if not faults:
                return value
            if attempt == attempts:
                self.log.warn("model.json_shape_accepted", task=task,
                              faults="; ".join(faults))
                return value
            self.log.warn("model.json_shape_retry", task=task, attempt=attempt,
                          faults="; ".join(faults))
            current = (
                f"{prompt}\n\nYour previous reply was valid JSON of the wrong "
                f"shape: {'; '.join(faults)}. Send the whole reply again, in "
                "the shape asked for above."
            )
        raise ModelError(f"task {task}: no parseable JSON after {attempts} attempts: {last_error}")

    def routed_tasks(self) -> list[str]:
        """Task kinds some agent actually asks for.

        Imported here rather than at module scope because agents.py imports
        this module. Checking the full task list instead would fail a run over
        a model that nothing was ever going to call.
        """
        from .agents import ROUTING
        return sorted(set(ROUTING.values()))

    def preflight(self) -> list[str]:
        """Verify every model an agent routes to is actually installed locally."""
        report = []
        for task in self.routed_tasks():
            backend = self.backend(task)
            name = backend.cfg.model
            try:
                installed = backend.available_models()
            except ModelError as exc:
                raise ConfigError(
                    f"task {task!r}: inference server {backend.cfg.base_url} unreachable. "
                    f"Start it before running Hermes. ({exc})"
                )
            if installed and not _model_present(name, installed):
                raise ConfigError(
                    f"task {task!r}: model {name!r} is not installed on "
                    f"{backend.cfg.base_url}. Installed: {', '.join(installed) or 'none'}"
                )
            report.append(f"{task}: {name} @ {backend.cfg.base_url} ok")
        distinct = {self.backend(t).cfg.model for t in self.routed_tasks()}
        if len(distinct) > 1:
            # Routing roles to different models only pays off if they are
            # co-resident; otherwise every role switch is a full model load,
            # and the loop switches roles several times per ticket.
            self.log.warn("model.multiple_models", count=len(distinct),
                          models=", ".join(sorted(distinct)),
                          hint="keep them loaded together or route every role "
                               "to one model")
        return report


#: How local servers say the prompt does not fit the loaded context: vLLM's
#: "maximum context length is N tokens", llama.cpp's "exceeds the available
#: context size", and kin.
_CONTEXT_LIMIT = re.compile(
    r"context (?:length|size|window)|max_model_len|max_tokens|token limit"
    r"|too many tokens", re.IGNORECASE)

#: How they name the one optional field Hermes adds: ``response_format`` on an
#: OpenAI-compatible server, ``format`` on ollama.
_PARAMETER_REFUSAL = re.compile(
    r"response_format|json_object|json_schema|guided[_ ]|structured output"
    r"|\bformat\b", re.IGNORECASE)


def _rejects_parameter(exc: ModelError) -> bool:
    """True when a local server refused the *structured-output flag* itself.

    A 4xx alone does not mean that. vLLM answers HTTP 400 when the prompt
    exceeds ``--max-model-len``, which is a routine outcome on the repair path
    where journal, file listing, sources and failure output share one prompt.
    Reading that as a missing feature would disable constrained decoding for
    the rest of the run over a prompt that was merely too long, so the refusal
    has to name the parameter Hermes added before we believe it. A 5xx or a
    connection failure is a real fault and must surface either way.
    """
    text = str(exc)
    if " HTTP 4" not in text or _CONTEXT_LIMIT.search(text):
        return False
    return bool(_PARAMETER_REFUSAL.search(text))


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _model_present(name: str, installed: list[str]) -> bool:
    """Ollama reports ``qwen3:30b-a3b``; llama.cpp may report a path or alias."""
    if name in installed:
        return True
    stem = name.split(":")[0].split("/")[-1]
    return any(stem in candidate or candidate in name for candidate in installed)
