"""Resource-aware admission control.

Section 19: GPU-heavy inference, CPU-heavy benchmarking and memory-heavy
compilation must not run at the same time when the host cannot carry both.
:func:`~hermes.sandbox.probe_resources` has always reported what the machine
has; this is what finally reads it.

Two things ask permission here, and they are the two the factory actually
overlaps:

*Prefetched generation.* Generating the next ticket while the current one runs
its gates is deliberate -- the GPU would otherwise idle through minutes of
CPU -- but the foreground is free to make its own model call (a review, a
diagnosis, a repair) while the prefetch is still running. That is a second
concurrent generation, and a second generation costs VRAM that the first one
has not released. When the headroom is not there the prefetch is refused, and
the loop simply generates in sequence as it did before.

*Benchmark measurement.* The evolution engine's timings decide which candidate
survives, and a benchmark sharing the machine with an inference call measures
the contention rather than the candidate. A measurement therefore waits for
inference to drain and refuses new concurrent work for its duration.

The governor never blocks the factory's own foreground work. Everything it can
say is "not concurrently, not right now", which costs latency and never
correctness.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from .log import Logger, NULL
from .sandbox import probe_resources


class Governor:
    """Decides whether a second heavy job may start on this machine."""

    def __init__(self, cfg: dict | None = None, log: Logger = NULL, probe=probe_resources):
        self.cfg = cfg or {}
        self.log = log
        self._probe = probe
        self._cond = threading.Condition()
        #: Model calls currently running, foreground and background alike.
        self._inflight = 0
        #: Benchmarks currently being timed for a selection decision.
        self._measuring = 0
        self._cached: dict | None = None
        self._cached_at = 0.0
        #: Why the last request was refused, for the log and the tests.
        self.last_refusal = ""

    # -- the host --------------------------------------------------------
    def snapshot(self, max_age: float = 2.0) -> dict:
        """The probe, re-read at most every *max_age* seconds.

        Free memory moves under us, so a value cached for the whole run would
        be a fiction; re-probing per decision would shell out to nvidia-smi
        several times a ticket for a number that cannot meaningfully change in
        that time.
        """
        now = time.time()
        if self._cached is None or now - self._cached_at > max_age:
            try:
                self._cached = self._probe()
            except Exception as exc:  # noqa: BLE001 - a blind governor still admits
                self.log.warn("resources.probe_failed", error=str(exc)[:200])
                self._cached = {}
            self._cached_at = now
        return self._cached

    def headroom(self) -> tuple[bool, str]:
        """Whether the host has room for one more concurrent generation.

        A GPU is asked about its free VRAM; a host without one is asked about
        free RAM, because that is where the weights are then sitting. An
        unanswerable probe admits: refusing work because a measurement is
        missing would turn a machine without nvidia-smi into one that never
        prefetches.
        """
        info = self.snapshot()
        gpus = info.get("gpus") or []
        if gpus:
            reserve = float(self.cfg.get("reserve_vram_mb", 1024))
            free = max((_number(g.get("vram_free_mb")) for g in gpus), default=None)
            if free is None:
                return True, ""
            if free < reserve:
                return False, f"{free:.0f}MB VRAM free, {reserve:.0f}MB reserved"
            return True, ""
        free_ram = info.get("ram_free_gb")
        reserve_ram = float(self.cfg.get("reserve_ram_gb", 2))
        if free_ram is not None and float(free_ram) < reserve_ram:
            return False, f"{free_ram}GB RAM free, {reserve_ram}GB reserved"
        return True, ""

    # -- admission -------------------------------------------------------
    def admit_inference(self) -> bool:
        """May another model call start alongside the work already running?

        Called before submitting a prefetch. The answer is advisory in one
        direction only: a refusal means the work is done in sequence instead,
        never that it is dropped.
        """
        with self._cond:
            if self._measuring:
                return self._refuse("a benchmark is being measured")
            limit = int(self.cfg.get("max_concurrent_inference", 2))
            if self._inflight >= max(1, limit):
                return self._refuse(f"{self._inflight} model call(s) already running "
                                    f"(limit {limit})")
        ok, why = self.headroom()
        if not ok:
            with self._cond:
                return self._refuse(why)
        self.last_refusal = ""
        return True

    def _refuse(self, reason: str) -> bool:
        self.last_refusal = reason
        self.log.info("resources.refused", work="inference", reason=reason)
        return False

    @contextmanager
    def inference(self):
        """Register a model call for its duration."""
        with self._cond:
            self._inflight += 1
        try:
            yield
        finally:
            with self._cond:
                self._inflight -= 1
                self._cond.notify_all()

    @contextmanager
    def measuring(self):
        """Time a benchmark with the machine as quiet as it can be made.

        Waits for inference to drain, then holds off concurrent admissions
        until the measurement is over. The wait is bounded: a local model can
        take minutes on one call, and a benchmark blocked behind it forever
        would be a worse failure than a slightly noisy number.
        """
        deadline = time.time() + float(self.cfg.get("measure_drain_seconds", 120))
        with self._cond:
            while self._inflight and time.time() < deadline:
                self._cond.wait(timeout=max(0.0, deadline - time.time()))
            if self._inflight:
                self.log.warn("resources.measure_contended", inflight=self._inflight,
                              hint="benchmark timed while a model call was running")
            self._measuring += 1
        try:
            yield
        finally:
            with self._cond:
                self._measuring -= 1
                self._cond.notify_all()

    # -- observability ---------------------------------------------------
    def status(self) -> dict:
        with self._cond:
            return {"inflight": self._inflight, "measuring": self._measuring,
                    "last_refusal": self.last_refusal}


def _number(value) -> float | None:
    """nvidia-smi reports through a CSV, so every field arrives as a string."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
