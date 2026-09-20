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
from typing import Any, Iterable

from .config import Config, ConfigError, ModelConfig
from .log import Logger, NULL

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class ModelError(RuntimeError):
    pass


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
        #: Constrained decoding, until a server tells us it does not speak it.
        self.json_mode = cfg.structured_output

    def complete(self, system: str, prompt: str, *, temperature: float | None = None,
                 max_tokens: int | None = None, json_mode: bool = False) -> str:
        raise NotImplementedError

    def available_models(self) -> list[str]:
        raise NotImplementedError


class OllamaBackend(Backend):
    def complete(self, system, prompt, *, temperature=None, max_tokens=None,
                 json_mode=False):
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
        if json_mode and self.json_mode:
            payload["format"] = "json"
        data = _post(f"{self.cfg.base_url.rstrip('/')}/api/chat", payload, self.cfg.timeout)
        return (data.get("message") or {}).get("content", "")

    def available_models(self):
        data = _get(f"{self.cfg.base_url.rstrip('/')}/api/tags")
        return [m["name"] for m in data.get("models", [])]


class OpenAICompatBackend(Backend):
    """llama.cpp server, vLLM, and anything else speaking /v1/chat/completions."""

    def _base(self) -> str:
        base = self.cfg.base_url.rstrip("/")
        return base if base.endswith("/v1") else base + "/v1"

    def complete(self, system, prompt, *, temperature=None, max_tokens=None,
                 json_mode=False):
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
        if json_mode and self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = _post(f"{self._base()}/chat/completions", payload, self.cfg.timeout)
        choices = data.get("choices") or []
        if not choices:
            raise ModelError(f"empty completion from {self.cfg.base_url}")
        return choices[0].get("message", {}).get("content", "")

    def available_models(self):
        data = _get(f"{self._base()}/models")
        return [m["id"] for m in data.get("data", [])]


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

    def complete(self, task: str, system: str, prompt: str, **kw: Any) -> str:
        backend = self.backend(task)
        started = time.time()
        with self.governor.inference() if self.governor else nullcontext():
            text = backend.complete(system, prompt, **kw)
        elapsed = time.time() - started
        self.usage.append(
            {"task": task, "model": backend.cfg.model, "seconds": round(elapsed, 2),
             "chars_in": len(system) + len(prompt), "chars_out": len(text)}
        )
        self.log.debug("model.complete", task=task, model=backend.cfg.model,
                       seconds=round(elapsed, 1), chars_out=len(text))
        return strip_reasoning(text)

    def _json_attempt(self, task: str, system: str, prompt: str, kw: dict) -> str:
        """One completion asked for under constrained decoding where supported."""
        backend = self.backend(task)
        try:
            return self.complete(task, system, prompt, json_mode=True, **kw)
        except ModelError as exc:
            if not (backend.json_mode and _rejects_parameter(exc)):
                raise
            # The server does not speak this parameter. Losing the guarantee is
            # survivable; failing the task over a capability probe is not.
            backend.json_mode = False
            self.log.warn("model.json_mode_unsupported", task=task, error=str(exc)[:200])
            return self.complete(task, system, prompt, **kw)

    def complete_json(self, task: str, system: str, prompt: str, *, attempts: int = 3,
                      **kw: Any) -> Any:
        """Completion that must parse as JSON. Retries with a repair nudge.

        Constrained decoding makes malformed JSON structurally impossible on
        servers that support it, so the repair loop is the fallback for the
        ones that do not rather than the normal path. Each repair costs a whole
        generation, which on a local model is the most expensive thing here.
        """
        last_error = ""
        current = prompt
        for attempt in range(1, attempts + 1):
            text = self._json_attempt(
                task, system + "\n\nRespond with JSON only. No prose.", current, kw)
            try:
                return extract_json(text)
            except ValueError as exc:
                last_error = str(exc)
                self.log.warn("model.json_retry", task=task, attempt=attempt)
                current = (
                    f"{prompt}\n\nYour previous reply was not valid JSON ({exc}). "
                    "Reply with a single valid JSON value and nothing else."
                )
                kw["temperature"] = min(0.8, kw.get("temperature", 0.2) + 0.2 * attempt)
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


def _model_present(name: str, installed: list[str]) -> bool:
    """Ollama reports ``qwen3:30b-a3b``; llama.cpp may report a path or alias."""
    if name in installed:
        return True
    stem = name.split(":")[0].split("/")[-1]
    return any(stem in candidate or candidate in name for candidate in installed)
