"""Bringing the local inference server up for the length of a run.

``start.sh`` writes a systemd user unit for the vLLM server but leaves it
disabled on purpose. The server holds the entire model in VRAM for as long as
it lives — vLLM has no CPU offload and never evicts — so a machine that boots
straight into it has given several gigabytes of its GPU to a process that may
sit idle for days.

The factory is the only thing that needs the server, and it knows exactly when:
a project starting or continuing. So Hermes starts the unit itself at the top of
a run, waits for the weights to load, and stops it again when the run ends —
but only if it was the one that started it. A server you brought up by hand
stays up, because you did that for a reason.

Nothing here is required: no unit, no systemd, or an autostart the operator
turned off all fall through quietly and leave
:meth:`hermes.models.Router.preflight` to report an unreachable server in the
detail it already does.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from contextlib import contextmanager

from .config import Config, is_local_url
from .log import NULL, Logger
from .models import Router

# -- what the unit serves with -----------------------------------------------

#: How the n-gram proposer drafts. It looks for the last ``prompt_lookup_min``
#: tokens of the generation somewhere earlier in the prompt and, on a hit,
#: offers whatever followed them there as the next ``num_speculative_tokens``.
#: The minimum is the knob that matters: below three the matches are noise, and
#: high above it the copied runs this factory produces are missed entirely.
NGRAM = {"method": "ngram", "num_speculative_tokens": 5,
         "prompt_lookup_min": 3, "prompt_lookup_max": 8}

#: First vLLM release that takes the dictionary above as ``--speculative-config``.
#: Older ones spell the same thing across four separate flags.
SPECULATIVE_SINCE = (0, 9)


def serve_flags(config: Config, vllm_version: str = "", log: Logger = NULL) -> list[str]:
    """Optional ``vllm serve`` flags for the unit ``start.sh`` writes.

    Each item is one continuation line of the unit's ``ExecStart``, not one
    argv word: the speculation settings are a JSON document that has to reach
    vLLM whole.

    Two flags, for two properties of this workload that are not true of
    serving in general.

    *Prefix caching* pays because the factory asks one model about one project
    dozens of times, and :class:`hermes.agents.Prompt` assembles every one of
    those calls stable-first — identity and architecture, then the ticket, then
    the volatile tail of failures and repair instructions. The shared prefix is
    therefore long and genuinely shared, which is the condition the cache needs
    and the reason the block order is not free to change. Recent vLLM enables
    it anyway; the flag is written because the unit is also the record of what
    is being relied on, and an older engine has it off.

    *Speculation* pays because most of what this factory decodes is text it was
    just shown: a search/replace ``find`` copied character for character out of
    the file in the prompt, a whole-file rewrite that re-emits the file around
    a small change, the fixed keys of the change protocol. The n-gram proposer
    is chosen over a draft model deliberately — ``start.sh`` warns at cards
    under 10 GiB, and a draft model wants VRAM that such a card does not have,
    while n-gram has no weights at all. Its miss case is cheap in the same way:
    no match in the prompt means no draft proposed, so the cost is a lookup
    rather than a forward pass spent on tokens that get rejected.

    A vLLM old enough to want the four-flag spelling is given no speculation
    rather than a second spelling to keep correct. A version that cannot be
    read at all is treated as new enough, because ``start.sh`` rewrites the
    unit without these flags and restarts when the server refuses them — which
    is the same safety net that makes defaulting them on defensible.
    """
    spec = config.get("server", {}) or {}
    flags = []
    if spec.get("prefix_caching", True):
        flags.append("--enable-prefix-caching")

    method = spec.get("speculative", "ngram")
    if isinstance(method, bool):
        # YAML reads a bare `off` as False, so the spelling that turns this
        # off arrives here as a boolean; and someone who writes `true` means
        # the method there is, not an unknown one.
        method = "ngram" if method else "off"
    method = str(method or "off").strip().lower()
    if method in ("off", "none", "false", ""):
        return flags
    if method != "ngram":
        log.warn("server.speculative_unknown", method=method)
        return flags

    release = _release(vllm_version)
    if release and release < SPECULATIVE_SINCE:
        log.info("server.speculative_unsupported", version=vllm_version,
                 since=".".join(str(part) for part in SPECULATIVE_SINCE))
        return flags
    flags.append("--speculative-config " + _quote(json.dumps(NGRAM, separators=(",", ":"))))
    return flags


def _release(version: str) -> tuple[int, int] | None:
    """Major and minor of a vLLM version string, or None if it says nothing."""
    match = re.match(r"\s*(\d+)\.(\d+)", version or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def _quote(value: str) -> str:
    """Protect *value* from systemd's own tokeniser.

    systemd splits ``ExecStart`` itself and strips double quotes as it goes,
    which would hand vLLM ``{method:ngram}`` — valid to a shell reader, not to
    a JSON parser. Single quotes survive whatever they wrap.
    """
    if "'" in value:
        raise ValueError(f"cannot put {value!r} in a unit file: it contains a quote")
    return f"'{value}'"


def _systemctl(args: list[str]) -> tuple[int, str]:
    """Run ``systemctl --user`` and return its exit code and merged output."""
    if not shutil.which("systemctl"):
        return 127, "systemctl not found"
    try:
        proc = subprocess.run(["systemctl", "--user", *args],
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 124, str(exc)
    return proc.returncode, proc.stdout + proc.stderr


def unit_state(unit: str, run=None) -> str:
    """One word for how *unit* is doing, for ``hermes doctor``.

    "inactive" is the expected answer between runs, not a fault: the unit is
    installed and Hermes starts it when a project does.
    """
    run = run or _systemctl
    if run(["cat", unit])[0] != 0:
        return "not installed"
    code, out = run(["is-active", unit])
    state = out.strip().splitlines()[-1].strip() if out.strip() else ""
    return state or f"unknown (exit {code})"


class ModelServer:
    """The local inference server, started on demand for one run."""

    def __init__(self, unit: str, *, wait: int = 600, stop_after_run: bool = True,
                 log: Logger = NULL, run=None, sleep=time.sleep):
        self.unit = unit
        self.wait = int(wait)
        self.stop_after_run = bool(stop_after_run)
        self.log = log
        # Resolved per call rather than bound as a default, so a test can
        # replace the module-level runner and still exercise from_config.
        self._run = run or (lambda args: _systemctl(args))
        self._sleep = sleep
        #: True only when this process started the unit, which is the one case
        #: in which it may also stop it.
        self.started = False

    @classmethod
    def from_config(cls, config: Config, log: Logger = NULL) -> "ModelServer | None":
        spec = config.get("server", {}) or {}
        unit = str(spec.get("unit") or "").strip()
        if not spec.get("autostart", True) or not unit:
            return None
        return cls(unit,
                   wait=int(spec.get("wait", 600)),
                   stop_after_run=bool(spec.get("stop_after_run", True)),
                   log=log)

    # -- lifecycle -----------------------------------------------------------

    def ensure(self, router: Router) -> None:
        """Start the unit if a routed local endpoint is not answering.

        Never raises. ``preflight`` runs immediately after this and is the one
        place allowed to fail a run over a server that is not there; whatever
        went wrong here, its message names the endpoint and the model and is
        the better one to show.
        """
        down = [task for task in _local_tasks(router) if not _reachable(router, task)]
        if not down:
            return
        code, _ = self._run(["cat", self.unit])
        if code != 0:
            # No such unit: the operator runs the server some other way.
            self.log.info("server.no_unit", unit=self.unit)
            return
        self.log.info("server.start", unit=self.unit, tasks=",".join(sorted(down)))
        code, out = self._run(["start", self.unit])
        if code != 0:
            self.log.warn("server.start_failed", unit=self.unit,
                          error=out.strip()[-400:] or f"exit {code}")
            return
        self.started = True
        self._await(router, down)

    def release(self) -> None:
        """Stop the unit, if this run is what brought it up."""
        if not (self.started and self.stop_after_run):
            return
        self.started = False
        code, out = self._run(["stop", self.unit])
        if code != 0:
            self.log.warn("server.stop_failed", unit=self.unit,
                          error=out.strip()[-400:] or f"exit {code}")
            return
        self.log.info("server.stopped", unit=self.unit)

    def __enter__(self) -> "ModelServer":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # -- readiness -----------------------------------------------------------

    def _await(self, router: Router, tasks: list[str]) -> None:
        """Wait for every endpoint in *tasks* to answer, up to ``wait`` seconds.

        vLLM accepts the port only once the weights are on the GPU, which takes
        minutes for a multi-gigabyte model, so a run that starts the unit and
        proceeds immediately would fail preflight on a server that was seconds
        from being ready.
        """
        started = time.monotonic()
        announced = 0.0
        while True:
            waiting = [task for task in tasks if not _reachable(router, task)]
            elapsed = time.monotonic() - started
            if not waiting:
                self.log.info("server.ready", unit=self.unit, seconds=round(elapsed, 1))
                return
            if elapsed >= self.wait:
                self.log.warn("server.timeout", unit=self.unit, seconds=round(elapsed, 1),
                              waiting=",".join(sorted(waiting)))
                return
            if elapsed - announced >= 30:
                announced = elapsed
                self.log.info("server.loading", unit=self.unit,
                              seconds=round(elapsed), limit=self.wait)
            self._sleep(3)


@contextmanager
def running(config: Config, router: Router, log: Logger = NULL):
    """Hold the local server up for a block that talks to models.

    Every entry point that generates — a run, an optimisation, a release, a
    self-improvement — wraps its work in this, so "the server is up while a
    project is being worked on" holds however the work was started. Yields the
    :class:`ModelServer`, or None when there is nothing for Hermes to manage.
    """
    server = ModelServer.from_config(config, log)
    if server is None:
        yield None
        return
    server.ensure(router)
    try:
        yield server
    finally:
        server.release()


def _local_tasks(router: Router) -> list[str]:
    """Routed tasks served from this machine, one per distinct endpoint.

    A remote endpoint (only reachable with ``allow_remote``) is not ours to
    start, and two roles on one endpoint would otherwise be probed twice.
    """
    tasks, seen = [], set()
    try:
        routed = router.routed_tasks()
    except Exception:  # a router that cannot say what it routes starts nothing
        return []
    for task in routed:
        try:
            base_url = router.backend(task).cfg.base_url
        except Exception:  # a broken config is preflight's to report
            continue
        if base_url in seen or not is_local_url(base_url):
            continue
        seen.add(base_url)
        tasks.append(task)
    return tasks


def _reachable(router: Router, task: str) -> bool:
    """A probe, not a check: any failure here means "not answering yet"."""
    try:
        router.backend(task).available_models()
    except Exception:
        return False
    return True
