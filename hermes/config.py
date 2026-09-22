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

from .schema import LEVELS, OFF, SCHEMA

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
            "backend": "vllm",
            # install.sh serves one model here under a systemd user unit.
            "base_url": "http://localhost:6767/v1",
            "model": "qwen2.5-coder-7b-instruct-awq",
            # vLLM reserves the KV cache for the full context at startup, so
            # this is a VRAM commitment rather than a limit, and it must match
            # the --max-model-len the server was started with.
            "context": 16384,
            "temperature": 0.2,
            "timeout": 900,
            # Constrained decoding where supported: schema | json | off.
            "structured_output": SCHEMA,
            "keep_alive": "30m",         # ollama only; vLLM never evicts
        },
    },
    "routing": {role: "default" for role in TASK_ROLES},
    "server": {
        # install.sh writes this unit but leaves it disabled: the server pins
        # the whole model in VRAM, so it runs while a project runs and not
        # while the machine merely happens to be on.
        "autostart": True,
        "unit": "hermes-vllm.service",
        "wait": 600,             # seconds allowed for the weights to load
        "stop_after_run": True,  # only ever stops a server Hermes started
    },
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
        # Blank uses the default HTML search endpoint; set it to a SearXNG
        # instance you host to keep the lookup off a third party.
        "search": "",                 # e.g. http://searx.lan/search?q={query}
        "max_searches": 8,            # searches one run may make
        "max_fetches": 24,            # pages one run may retrieve, all rounds
        "max_fetches_per_round": 6,   # and what one round may spend of them
        "fetch_bytes": 250000,        # cap on a single retrieved page
        "fetch_timeout": 30,          # seconds one fetch may take
        "allow_private": False,       # reach a LAN/loopback documentation server
        # A hypothesis nobody measures is a note. These bound what settling
        # them costs: how many of a round's hypotheses become experiment
        # tickets, how long one experiment may run, how many times it is
        # measured (>1 is aggregated by median, for a timing experiment on a
        # noisy box), and how many tries it gets at a program that runs.
        "experiments": True,
        "experiments_per_round": 2,
        "experiment_timeout": 300,
        "experiment_repeats": 1,
        "experiment_attempts": 2,
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
        "files": 6,             # existing sources shown to the implementer
        # How a repeated failure is asked differently. One entry per attempt
        # after the first; each inherits the rung below it and changes only
        # what it names, and the top rung repeats once the list runs out.
        # Roles are deliberately the knob rather than models: routing a retry
        # to a second model reloads weights, which costs more than the retry
        # saves. [] disables escalation.
        "escalation": [
            {"candidates": 2, "files": 10},   # attempt 2: race more, read more
            {"role": "architect"},            # attempt 3: ask the reasoning role
        ],
        # What the factory remembers about its own retries, across every
        # project it has ever built: which rung actually rescued which kind
        # of failure. Advisory -- with no evidence the fixed ladder above
        # decides, exactly as it did before this existed. See hermes.ledger.
        "ledger": True,
        "ledger_path": "",      # blank uses $XDG_STATE_HOME/hermes/factory.db
        "edit_protocol": True,  # let agents patch a file instead of rewriting it
        "prefetch": True,       # generate the next ticket while this one verifies
        "tool_calls": 6,        # read-only workspace lookups one answer may make
        "tool_rounds": 3,       # extra generations those lookups may cost; 0 for
                                # neither, which is the single-shot prompt
    },
    "resources": {
        "reserve_ram_gb": 2,            # free RAM a host without a GPU keeps back
        "reserve_vram_mb": 1024,        # free VRAM a second concurrent generation needs
        "max_concurrent_inference": 2,  # 1 stops generation overlapping at all
        "measure_drain_seconds": 120,   # how long a benchmark waits for the GPU to clear
    },
    "verification": {
        # Mutation testing: break the code the ticket just wrote and see
        # whether its own tests go red. The factory writes both sides, so a
        # green suite is the weakest evidence it collects -- but the check
        # costs one full test run per injected fault, which is why it is off
        # until an operator decides the project is worth it. See
        # hermes.mutate.
        "mutation": {
            "enabled": False,
            "mutants": 5,      # faults injected per ticket
            "budget": 180,     # seconds the whole survey may spend
            "timeout": 120,    # seconds one test run under one fault may take
            "screen": True,    # ask a model whether a survivor is reachable at
                               # all; most survivors are faults no test could
                               # catch, and reporting those teaches nothing
            "block": True,     # a survivor the screen confirmed fails the
                               # ticket, which sends the concrete failing case
                               # back to the repair path. false only reports it
        },
    },
    "security": {
        # A second opinion on the security gate, for the defects no regex in
        # UNSAFE_PATTERNS expresses. Off by default and advisory when on: an
        # un-tuned checkpoint scores near chance, and a guessed defect that
        # fails a correct ticket costs more than the miss it prevented.
        # contrib/laya_server.py serves the endpoint this expects.
        "classifier": {
            "enabled": False,
            "base_url": "http://localhost:6768",
            "model": "laya",
            "timeout": 30,
            "threshold": 0.75,   # calibrate the checkpoint before trusting this
            "block": False,      # let a finding fail the gate, not just warn
            "max_chunks": 400,   # windows one scan may classify
        },
    },
    "git": {"enabled": True, "author": "Hermes Factory <hermes@localhost>"},
    "github": {
        "enabled": False, "private": True, "owner": None, "name": None,
        "early": True,        # create the repository once the board exists, so
                              # the projection can watch a running project. The
                              # code is pushed on the first ticket that passes.
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
    base_url: str = "http://localhost:6767/v1"
    context: int = 16384
    temperature: float = 0.2
    timeout: int = 900
    allow_remote: bool = False
    #: How hard the server is asked to constrain decoding: ``schema`` sends
    #: the per-task JSON Schema, ``json`` asks only for some JSON, ``off``
    #: asks for nothing. See :func:`structured_level`.
    structured_output: str = SCHEMA
    #: How long ollama should hold the weights between calls. Ignored by vLLM,
    #: which keeps the model resident for the lifetime of the process.
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
        if self.structured_output not in LEVELS:
            raise ConfigError(
                f"model '{self.role}': structured_output is "
                f"{self.structured_output!r}; it must be one of "
                f"{', '.join(LEVELS)} (or true/false for the outer two)."
            )


def structured_level(value: Any) -> str:
    """Read ``structured_output`` in either the old spelling or the new one.

    It was a boolean before schemas existed, and configs written then are
    still correct: ``true`` meant "constrain as hard as you can", which is now
    ``schema``, and ``false`` still means ask for nothing. An unrecognised
    value is passed through for :meth:`ModelConfig.validate` to reject by
    name, rather than being silently read as a truthy string.
    """
    if isinstance(value, bool):
        return SCHEMA if value else OFF
    return str(value).strip().lower()


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


def user_config_paths() -> list[Path]:
    """Where a per-user config may live, in precedence order.

    install.sh and uninstall.sh both address this file as
    ``${XDG_CONFIG_HOME:-$HOME/.config}/hermes/config.yaml``. Honouring only
    the second half of that would mean the installer writes the name of the
    model it is serving into a file Hermes never reads. ``~/.config`` stays in
    the list regardless, because installs predating this wrote there.
    """
    paths = []
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        paths.append(Path(xdg) / "hermes" / "config.yaml")
    fallback = Path.home() / ".config" / "hermes" / "config.yaml"
    if fallback not in paths:
        paths.append(fallback)
    return paths


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
            *user_config_paths(),
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
            backend=spec.get("backend", "vllm"),
            model=spec.get("model", ""),
            base_url=spec.get("base_url", "http://localhost:6767/v1"),
            context=int(spec.get("context", 16384)),
            temperature=float(spec.get("temperature", 0.2)),
            timeout=int(spec.get("timeout", 900)),
            allow_remote=bool(spec.get("allow_remote", False)),
            structured_output=structured_level(spec.get("structured_output", SCHEMA)),
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
#
# install.sh serves one model with vLLM on port 6767, as a systemd user unit:
# `systemctl --user status hermes-vllm.service`. Hermes starts that unit when
# a project starts or continues and stops it when the run ends, so the model
# does not sit in VRAM between runs. The model name below must be the one the
# server answers to, which is what `curl localhost:6767/v1/models` reports.
models:
  default:
    backend: vllm                    # vllm | openai | llamacpp | ollama
    base_url: http://localhost:6767/v1
    model: qwen2.5-coder-7b-instruct-awq
    context: 16384                   # must match the server's --max-model-len
    temperature: 0.2
    structured_output: schema        # schema | json | off (see README)
    keep_alive: 30m                  # ollama only; vLLM never evicts
  # coder:
  #   backend: ollama                # a second server, if you run one
  #   base_url: http://localhost:11434
  #   model: qwen2.5-coder:32b

routing:
  research: default
  architecture: default
  coding: default
  review: default
  simple: default
  evolution: default

server:
  autostart: true              # start the unit below when a run begins
  unit: hermes-vllm.service    # systemd *user* unit, so `systemctl --user`
  wait: 600                    # seconds allowed for the weights to load
  stop_after_run: true         # set false to leave the model resident between
                               # runs, or when two projects share one server:
                               # Hermes only ever stops one it started itself

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
                         # on: a search turns each question into URLs that
                         # exist, and the sandbox retrieves and cites them.
  max_rounds: 3
  search: ""             # blank uses the default HTML search endpoint; set it
                         # to http://searx.lan/search?q={query} for a SearXNG
                         # you host, with allow_private for one on your network
  max_searches: 8        # searches one run may make
  max_fetches: 24        # pages one run may retrieve, across every round and
                         # every ticket failure that escalates into research
  max_fetches_per_round: 6   # and what a single round may spend of them
  fetch_bytes: 250000    # cap on a single page
  allow_private: false   # only set for a documentation server you host
  experiments: true      # a hypothesis the round raises becomes an experiment
                         # ticket: a program is written, run in the sandbox,
                         # and its measurement settles the question. Off leaves
                         # the hypothesis as a note in docs/research/
  experiments_per_round: 2   # hypotheses from one round that get measured
  experiment_timeout: 300    # seconds one run of the program may take
  experiment_repeats: 1      # runs per experiment; >1 reports their median
  experiment_attempts: 2     # rewrites allowed when the program measures
                             # nothing it can report

execution:
  candidates: 1          # samples raced through the gates per ticket; >1 costs
                         # a whole extra verification pass, so raise it only if
                         # the gates are cheap relative to generation
  files: 6               # existing sources shown to the implementer
  escalation:            # how attempt 2, 3, ... are asked differently, each
                         # rung inheriting the one below it. The top rung
                         # repeats; [] retries identically instead
    - {candidates: 2, files: 10}   # attempt 2: race more samples, read more code
    - {role: architect}            # attempt 3: ask the reasoning role instead,
                                   # on the model already resident -- a rung
                                   # never reloads weights mid-ticket
  edit_protocol: true    # let agents patch a file instead of rewriting it whole
  prefetch: true         # generate the next ticket while this one verifies
  tool_calls: 6          # read-only lookups (read_file, grep, list_dir, and
                         # find_symbol/find_references/outline/recall where the
                         # context database has them) the implementer or
                         # reviewer may make before it answers, so it can check
                         # a name instead of inventing one
  tool_rounds: 3         # extra generations those lookups may cost. Set either
                         # to 0 to go back to a single-shot prompt

evolution:
  enabled: true
  max_generations: 20
  population: 3          # candidates proposed per generation
  baseline_repeats: 3    # runs used to measure benchmark noise
  min_relative_gain: 0.01
  screen_factor: 2.0     # kill a run this much slower than the incumbent

verification:
  mutation:              # measure the tests, rather than trusting the suite
    enabled: false       # that the same factory wrote to check its own code.
                         # Once a ticket passes every other gate, a small fault
                         # is injected into the code it touched and the tests
                         # are re-run: one the tests do not notice is a change
                         # that would have shipped. Costs a test run per fault
    mutants: 5           # faults injected per ticket
    budget: 180          # seconds the whole survey may spend
    timeout: 120         # seconds one test run under one fault may take
    screen: true         # ask a model whether a survivor is reachable at all;
                         # most are faults no test could ever have caught
    block: true          # a confirmed survivor fails the ticket, which sends
                         # the failing case to the repair path. false reports it

github:
  enabled: false
  private: true
  early: true            # create the repository when planning produces a board,
                         # so the issues track a project that is still running.
                         # The code is pushed on the first ticket that passes.
  issues: true           # mirror the ticket board onto issues. One way: the
                         # board is the source of truth and an edit made on
                         # GitHub is never read back.
"""
