"""The factory's evaluation suite: Hermes measuring Hermes.

Nothing here judges a generated project. It measures the factory itself, by
building the same fixed set of ideas the same way every time, so that a change
to prompts, routing, scheduling, the edit protocol or the sandbox can be shown
to have helped instead of asserted to have. Section 35 makes this a
prerequisite -- self-improvement that cannot measure itself is churn -- and it
is equally the only honest way to tune the knobs under ``execution``.

The suite runs headless and needs nothing but the locally installed models the
rest of Hermes already requires.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config

@dataclass
class Task:
    name: str
    idea: str
    #: Iteration ceiling. An entry allowed to run forever reports nothing when
    #: it stalls, and a stall is the failure mode worth catching.
    iterations: int = 40


#: Ideas small enough to finish on local models, chosen to differ in what they
#: stress: pure logic, file IO, parsing, mutable state, a measurable objective
#: (which is what exercises evolution) and a command line interface.
SUITE = (
    Task("json-diff", "Build a deterministic JSON diff tool in Python that "
                      "reports added, removed and changed keys."),
    Task("roman", "Build a Python library converting between Roman numerals and "
                  "integers, rejecting malformed numerals."),
    Task("csv-stats", "Build a Python command line tool that reads a CSV file and "
                      "prints per-column count, mean, min and max."),
    Task("ini-parser", "Build a Python parser for INI configuration files with "
                       "sections, comments and type coercion."),
    Task("lru-cache", "Build a Python LRU cache with a size bound, expiry and "
                      "hit/miss statistics."),
    Task("markdown-toc", "Build a Python tool that inserts a table of contents "
                         "into a markdown file from its headings."),
    Task("primes", "Build a Python prime sieve with a benchmark that prints "
                   "SCORE: <seconds> for sieving the first ten million integers.",
         iterations=50),
    Task("task-queue", "Build an in-memory Python task queue with priorities, "
                       "retries and a persistent JSON log."),
)


@dataclass
class Trial:
    """What one suite entry cost and what it produced."""

    name: str
    phase: str = "UNKNOWN"
    completed: bool = False
    tickets: int = 0
    tickets_completed: int = 0
    tickets_abandoned: int = 0
    #: Share of completed tickets that passed every gate on the first attempt.
    #: The single most sensitive number to prompt and protocol changes.
    first_pass: float = 0.0
    seconds: float = 0.0
    inference_seconds: float = 0.0
    sandbox_seconds: float = 0.0
    model_calls: int = 0
    chars_out: int = 0
    error: str = ""
    root: str = ""

    def row(self) -> str:
        mark = "pass" if self.completed else "FAIL"
        return (f"{self.name:<14} {mark:>4} {self.phase:<14} "
                f"tickets={self.tickets_completed}/{self.tickets} "
                f"first_pass={self.first_pass:.0%} "
                f"{self.seconds / 60:.1f}m "
                f"(gpu {self.inference_seconds / 60:.1f}m, "
                f"cpu {self.sandbox_seconds / 60:.1f}m)"
                + (f" error={self.error[:60]}" if self.error else ""))


def collect(name: str, state, board, usage: list[dict], sandbox_seconds: float,
            seconds: float, error: str = "", root: str = "") -> Trial:
    """Turn a finished run into numbers.

    Kept free of the factory so the accounting can be tested without standing
    up a project or calling a model.
    """
    counts = board.counts()
    done = board.by_status("completed")
    return Trial(
        name=name,
        phase=getattr(state, "phase", "UNKNOWN"),
        completed=getattr(state, "phase", "") == "COMPLETED",
        tickets=len(board.tickets),
        tickets_completed=counts.get("completed", 0),
        tickets_abandoned=counts.get("abandoned", 0),
        first_pass=(sum(1 for t in done if t.attempts <= 1) / len(done)) if done else 0.0,
        seconds=round(seconds, 1),
        inference_seconds=round(sum(c.get("seconds", 0.0) for c in usage), 1),
        sandbox_seconds=round(sandbox_seconds, 1),
        model_calls=len(usage),
        chars_out=sum(c.get("chars_out", 0) for c in usage),
        error=error,
        root=root,
    )


def run_task(task: Task, root: Path, config: Config | None = None,
             max_iterations: int | None = None, quiet: bool = True) -> Trial:
    """Build one suite entry from scratch and report what it cost."""
    from .factory import create
    from .state import State
    from .tickets import Board

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    factory = None
    error = ""
    try:
        factory = create(root, task.idea, config, quiet=quiet)
        factory.run(max_iterations=max_iterations or task.iterations)
    except Exception as exc:  # noqa: BLE001 - a crashed entry is a result, not a stop
        error = f"{type(exc).__name__}: {exc}"[:300]
    if factory is None:
        return collect(task.name, State(idea=task.idea), Board(root), [], 0.0,
                       time.time() - started, error, str(root))
    return collect(task.name, factory.state, factory.board,
                   list(getattr(factory.router, "usage", [])),
                   factory.sandbox.spent, time.time() - started, error, str(root))


def run_suite(root: Path, config: Config | None = None,
              names: list[str] | None = None, max_iterations: int | None = None,
              quiet: bool = True, on_trial=None) -> list[Trial]:
    """Run the suite serially.

    Serially on purpose: two projects sharing one GPU measure each other's
    queueing delay rather than the factory, and the wall-clock number is half
    the point of running this at all.
    """
    root = Path(root)
    chosen = [t for t in SUITE if not names or t.name in names]
    trials = []
    for task in chosen:
        trial = run_task(task, root / task.name, config, max_iterations, quiet)
        trials.append(trial)
        if on_trial is not None:
            on_trial(trial)
        write_report(root, trials)  # survive an interrupted suite
    return trials


def summarise(trials: list[Trial]) -> str:
    """The whole suite in a dozen lines, because a table nobody reads is a log.

    The GPU/CPU split is the line to watch when tuning: it says whether the
    factory is waiting on the model or on the code it wrote.
    """
    if not trials:
        return "no trials"
    seconds = sum(t.seconds for t in trials) or 1.0
    inference = sum(t.inference_seconds for t in trials)
    sandbox = sum(t.sandbox_seconds for t in trials)
    weight = sum(t.tickets_completed for t in trials)
    first = sum(t.first_pass * t.tickets_completed for t in trials)
    return "\n".join([t.row() for t in trials] + [
        "",
        f"completed      {sum(1 for t in trials if t.completed)}/{len(trials)}",
        f"first-pass     {(first / weight if weight else 0):.0%} of "
        f"{weight} completed tickets",
        f"wall clock     {seconds / 60:.1f}m",
        f"inference      {inference / 60:.1f}m ({inference / seconds:.0%})",
        f"sandbox        {sandbox / 60:.1f}m ({sandbox / seconds:.0%})",
    ])


def write_report(directory: Path, trials: list[Trial]) -> Path:
    """Write both machine-readable and human-readable results."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "trials": [asdict(t) for t in trials]}
    (directory / "results.json").write_text(json.dumps(payload, indent=2))
    report = directory / "report.md"
    report.write_text(
        "# Hermes evaluation\n\n"
        f"Run at {payload['at']}.\n\n```\n{summarise(trials)}\n```\n")
    return report


def compare(before: Path, after: Path) -> str:
    """Two result files side by side: the actual point of keeping them.

    A single run says nothing on its own; the question is always whether the
    change that came between these two runs helped.
    """
    def load(path: Path) -> dict[str, dict]:
        data = json.loads(Path(path).read_text())
        return {t["name"]: t for t in data.get("trials", [])}

    old, new = load(before), load(after)
    lines = [f"{'task':<14} {'completed':>18} {'first pass':>16} {'wall clock':>18}"]
    for name in sorted(set(old) | set(new)):
        a, b = old.get(name), new.get(name)
        if not a or not b:
            lines.append(f"{name:<14} {'only in one run':>18}")
            continue
        done = "{} -> {}".format(a["completed"], b["completed"])
        passed = "{:.0%} -> {:.0%}".format(a["first_pass"], b["first_pass"])
        clock = "{:.1f}m -> {:.1f}m".format(a["seconds"] / 60, b["seconds"] / 60)
        lines.append(f"{name:<14} {done:>18} {passed:>16} {clock:>18}")
    return "\n".join(lines)
