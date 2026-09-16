"""Bringing the local inference server up for the length of a run.

``install.sh`` writes a systemd user unit for the vLLM server but leaves it
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

import shutil
import subprocess
import time
from contextlib import contextmanager

from .config import Config, is_local_url
from .log import NULL, Logger
from .models import Router


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
