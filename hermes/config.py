"""Configuration loading and validation.

Hermes is local-model-only. Every inference endpoint is checked against
:data:`LOCAL_HOSTS` unless the operator explicitly opts out per role.
"""
from __future__ import annotations

import copy
import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}

#: Task kinds the router knows about. Each maps to a model role in config.
TASK_ROLES = (
    "research",
    "architecture",
    "coding",
    "review",
    "simple",
    "evolution",
)

DEFAULTS: dict[str, Any] = {
    "models": {
        "default": {
            "backend": "ollama",
            "base_url": "http://localhost:11434",
            "model": "qwen3:30b-a3b",
            "context": 32768,
            "temperature": 0.2,
            "timeout": 900,
            "structured_output": True,   # constrained decoding where supported
            "keep_alive": "30m",         # survive a long build without a reload
        },
    },
    "routing": {role: "default" for role in TASK_ROLES},
    "sandbox": {
        "runtime": "docker",          # docker | local
        "image": "python:3.11-slim",
        "network": "none",            # none | restricted | host
        "memory": "8g",
        "cpus": 4,
        "pids": 512,
        "timeout": 900,               # seconds per command
        "workdir": "/workspace",
    },
    "research": {
        "enabled": True,
        "network": False,             # allow network during research stage
        "max_rounds": 3,
        "max_unknowns": 6,
        "max_fetches": 6,             # pages one research round may retrieve
        "fetch_bytes": 250000,        # cap on a single retrieved page
        "fetch_timeout": 30,          # seconds one fetch may take
        "allow_private": False,       # reach a LAN/loopback documentation server
    },
    "evolution": {
        "enabled": True,
        "max_generations": 20,
        "population": 3,              # candidates proposed per generation
        "patience": 5,                # generations without improvement
        "baseline_repeats": 3,        # runs used to measure benchmark noise
        "min_relative_gain": 0.01,    # ignore gains smaller than this fraction
        "benchmark_timeout": 900,
        "screen_factor": 2.0,         # kill a run this much slower than the best
        "archive": 12,                # correct candidates kept as possible parents
    },
    "planning": {
        "max_tickets": 40,
        "max_decompose_depth": 2,
    },
    "execution": {
        "max_ticket_attempts": 3,
        "max_repair_attempts": 3,
        "max_loop_iterations": 400,
        "checkpoint_every": 1,
        "candidates": 1,        # independent samples raced through the gates
        "edit_protocol": True,  # let agents patch a file instead of rewriting it
        "prefetch": True,       # generate the next ticket while this one verifies
    },
    "resources": {
        "reserve_ram_gb": 2,            # free RAM a host without a GPU keeps back
        "reserve_vram_mb": 1024,        # free VRAM a second concurrent generation needs
        "max_concurrent_inference": 2,  # 1 stops generation overlapping at all
        "measure_drain_seconds": 120,   # how long a benchmark waits for the GPU to clear
    },
    "git": {"enabled": True, "author": "Hermes Factory <hermes@localhost>"},
    "github": {
        "enabled": False, "private": True, "owner": None, "name": None,
        "issues": True,       # project the ticket board onto issues, one way
        "issue_batch": 20,    # issues touched per save, so a freshly planned
                              # board spreads over iterations instead of
                              # spending a hundred API calls at once
    },
    "openhands": {"enabled": False, "command": "openhands"},
    "logging": {"level": "info"},
}


class ConfigError(RuntimeError):
    pass


@dataclass
class ModelConfig:
    role: str
    backend: str
    model: str
    base_url: str = "http://localhost:11434"
    context: int = 32768
    temperature: float = 0.2
    timeout: int = 900
    allow_remote: bool = False
    #: Ask the server to constrain decoding to JSON where it supports it.
    structured_output: bool = True
    #: How long ollama should hold the weights between calls.
    keep_alive: str = "30m"
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.backend not in ("ollama", "openai", "llamacpp", "vllm"):
            raise ConfigError(f"model '{self.role}': unknown backend {self.backend!r}")
        if not self.model:
            raise ConfigError(f"model '{self.role}': no model name/path configured")
        if not self.allow_remote and not is_local_url(self.base_url):
            raise ConfigError(
                f"model '{self.role}': {self.base_url} is not a local endpoint. "
                "Hermes refuses hosted inference APIs; set allow_remote: true only "
                "for a self-hosted server on your own network."
            )


def is_local_url(url: str) -> bool:
    """True when *url* points at loopback or a private/link-local address."""
    host = urlparse(url).hostname or ""
    if host in LOCAL_HOSTS:
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private or addr.is_link_local


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class Config:
    data: dict[str, Any]
    path: Path | None = None

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        candidates = [Path(path)] if path else [
            Path.cwd() / "hermes.yaml",
            Path.home() / ".config" / "hermes" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                loaded = yaml.safe_load(candidate.read_text()) or {}
                if not isinstance(loaded, dict):
                    raise ConfigError(f"{candidate}: top level must be a mapping")
                return cls(deep_merge(DEFAULTS, loaded), candidate)
        if path:
            raise ConfigError(f"config file not found: {path}")
        return cls(copy.deepcopy(DEFAULTS), None)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def model_for(self, task: str) -> ModelConfig:
        """Resolve a task kind to a concrete :class:`ModelConfig`."""
        models = self.get("models", {})
        name = self.get(f"routing.{task}") or "default"
        spec = models.get(name) or models.get("default")
        if spec is None:
            raise ConfigError(
                f"no model configured for task {task!r} (routing -> {name!r})"
            )
        known = {"backend", "model", "base_url", "context", "temperature",
                 "timeout", "allow_remote", "structured_output", "keep_alive"}
        cfg = ModelConfig(
            role=name,
            backend=spec.get("backend", "ollama"),
            model=spec.get("model", ""),
            base_url=spec.get("base_url", "http://localhost:11434"),
            context=int(spec.get("context", 32768)),
            temperature=float(spec.get("temperature", 0.2)),
            timeout=int(spec.get("timeout", 900)),
            allow_remote=bool(spec.get("allow_remote", False)),
            structured_output=bool(spec.get("structured_output", True)),
            keep_alive=str(spec.get("keep_alive", "30m")),
            extra={k: v for k, v in spec.items() if k not in known},
        )
        cfg.validate()
        return cfg

    def validate(self) -> list[str]:
        """Validate every routed model. Returns the list of role names checked."""
        seen = []
        for task in TASK_ROLES:
            cfg = self.model_for(task)
            seen.append(f"{task} -> {cfg.role}:{cfg.model} ({cfg.backend})")
        return seen

    def dump(self) -> str:
        return yaml.safe_dump(self.data, sort_keys=False)


SAMPLE_CONFIG = """\
# Hermes Project Factory configuration. Local models only.
models:
  default:
    backend: ollama                  # ollama | openai | llamacpp | vllm
    base_url: http://localhost:11434
    model: qwen3:30b-a3b
    context: 32768
    temperature: 0.2
    structured_output: true          # constrain decoding to JSON server-side
    keep_alive: 30m                   # do not evict the model during a build
  # coder:
  #   backend: openai                # llama.cpp / vLLM OpenAI-compatible server
  #   base_url: http://localhost:8080/v1
  #   model: /models/qwen2.5-coder-32b.gguf

routing:
  research: default
  architecture: default
  coding: default
  review: default
  simple: default
  evolution: default

sandbox:
  runtime: docker                    # docker | local
  image: python:3.11-slim
  network: none
  memory: 8g
  cpus: 4
  timeout: 900

research:
  enabled: true
  network: false         # off: findings come from what the local model knows.
                         # on: the sandbox retrieves and cites real pages.
  max_rounds: 3
  max_fetches: 6         # pages one research round may retrieve
  fetch_bytes: 250000    # cap on a single page
  allow_private: false   # only set for a documentation server you host

execution:
  candidates: 1          # samples raced through the gates per ticket; >1 costs
                         # a whole extra verification pass, so raise it only if
                         # the gates are cheap relative to generation
  edit_protocol: true    # let agents patch a file instead of rewriting it whole
  prefetch: true         # generate the next ticket while this one verifies

evolution:
  enabled: true
  max_generations: 20
  population: 3          # candidates proposed per generation
  baseline_repeats: 3    # runs used to measure benchmark noise
  min_relative_gain: 0.01
  screen_factor: 2.0     # kill a run this much slower than the incumbent

github:
  enabled: false
  private: true
  issues: true           # mirror the ticket board onto issues. One way: the
                         # board is the source of truth and an edit made on
                         # GitHub is never read back.
"""
