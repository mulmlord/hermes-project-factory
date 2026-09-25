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

import ast
import json
import shlex
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ElementTree
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config
from .verify import Gate, SOURCE_SUFFIXES, Verification, run_commands

#: Requirements documents and the held-out tests that score them, one folder
#: per library task. Shipped with the package and protected from
#: ``hermes improve``, because they are the instrument, not the subject.
SUITE_DIR = Path(__file__).with_name("suite")

#: Where the held-out tests are dropped inside the scored copy. Not a dot
#: directory, so pytest names the tests after it rather than after nothing.
HOLDOUT_DIR = "hermes_holdout"

@dataclass
class Task:
    name: str
    idea: str
    #: Iteration ceiling. An entry allowed to run forever reports nothing when
    #: it stalls, and a stall is the failure mode worth catching.
    iterations: int = 40
    #: Wall-clock ceiling. Iterations bound how many phases run, not how long
    #: one takes: a single stuck command is bounded only by the sandbox
    #: timeout, so without this one bad idea can consume a whole suite run.
    budget_seconds: float = 1800.0
    #: Folder under :data:`SUITE_DIR` whose tests score the finished project.
    #: Empty for the small ideas, which are judged only by whether the
    #: factory says it finished.
    holdout: str = ""

    @property
    def summary(self) -> str:
        return self.idea.splitlines()[0].lstrip("# ").strip()


def library(name: str) -> Task:
    """A task shaped like NL2Repo-Bench: a requirements document, an empty
    workspace, and tests the factory never sees.

    The document is the whole idea -- it pins the package, its modules and
    its public API, because a held-out suite can only import what was named.
    Scoring is the share of held-out tests that pass, which is dense where
    ``completed`` is a single bit, and independent of the factory's own
    verdict, which is the point: the failure these tasks exist to catch is a
    factory that stops early and calls a half-built library done.
    """
    return Task(name, (SUITE_DIR / name / "REQUIREMENTS.md").read_text(),
                iterations=120, budget_seconds=5400.0, holdout=name)


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
    # Multi-module libraries from a requirements document: date arithmetic
    # and a small grammar, a lexer-parser-renderer pipeline, and a data model
    # whose invariants span modules.
    library("cronspec"),
    library("stencil"),
    library("tally"),
)


@dataclass
class Holdout:
    """How a finished workspace did against tests it never saw."""

    passed: int = 0
    total: int = 0
    #: Why the score is not a measurement of the project, when it is not:
    #: a suite that could not run says nothing about the code it would test.
    note: str = ""


def expected_tests(folder: Path) -> dict[tuple[str, str], int]:
    """Cases per held-out test function, read from the source.

    The denominator comes from here rather than from what pytest reports,
    because a project that fails to import collects nothing, and a rate
    measured against what was collected would score that project 0/0.
    Only module-level functions count, and a parametrize whose cases are not
    a literal list is refused rather than guessed at.
    """
    counts: dict[tuple[str, str], int] = {}
    for path in sorted(Path(folder).glob("test_*.py")):
        for node in ast.parse(path.read_text()).body:
            if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
                continue
            cases = 1
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Call)
                        and getattr(decorator.func, "attr", "") == "parametrize"):
                    values = decorator.args[1] if len(decorator.args) > 1 else None
                    if not isinstance(values, (ast.List, ast.Tuple)):
                        raise ValueError(f"{path.name}:{node.name}: parametrize "
                                         "cases must be a literal list")
                    cases *= len(values.elts)
            counts[(path.stem, node.name)] = cases
    return counts


def passed_tests(junit: str, expected: dict[tuple[str, str], int]) -> int:
    """Count passing cases in a JUnit report, and only the ones we wrote.

    A case is keyed by its module and function, so a test the project
    smuggled in beside ours adds nothing, and no function is credited more
    cases than its source declares.
    """
    try:
        root = ElementTree.fromstring(junit)
    except ElementTree.ParseError:
        return 0
    credited: dict[tuple[str, str], int] = {}
    for case in root.iter("testcase"):
        if any(child.tag in ("failure", "error", "skipped") for child in case):
            continue
        key = (case.get("classname", "").rsplit(".", 1)[-1],
               case.get("name", "").split("[", 1)[0])
        if key in expected:
            credited[key] = min(expected[key], credited.get(key, 0) + 1)
    return sum(credited.values())


def score(task: Task, root: Path, sandbox_cfg: dict, setup: str = "") -> Holdout:
    """Run the held-out tests against a pristine export of the workspace.

    The export is what a release would ship, so a file the factory wrote and
    never committed does not count. The run is the generated project's code,
    so it happens in a fresh sandbox, and pytest is told to ignore every
    ``conftest.py``, the project's own pytest configuration and any plugin
    the project installed: those are the three ways a project can change
    what a test run reports without changing what its code does.
    """
    from .release import export
    from .sandbox import Sandbox

    folder = SUITE_DIR / task.holdout
    expected = expected_tests(folder)
    total = sum(expected.values())
    root = Path(root)
    if not root.is_dir():
        return Holdout(0, total, "no workspace to score")
    staging = Path(tempfile.mkdtemp(prefix="hermes-holdout-", dir=root.parent))
    try:
        export(root, staging)
        target = staging / HOLDOUT_DIR
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir()
        files = sorted(folder.glob("test_*.py"))
        for path in files:
            shutil.copy2(path, target / path.name)
        (target / "pytest.ini").write_text("[pytest]\n")
        box = Sandbox(staging, dict(sandbox_cfg))
        # The host's interpreter when the box is the host, since a bare
        # ``python`` there may be one without pytest.
        python = "python" if box.runtime == "docker" else shlex.quote(sys.executable)
        with box:
            if setup.strip():
                box.install(setup)
            if not box.run(f"{python} -m pytest --version").ok:
                if box.runtime != "docker" or not box.install(
                        "pip install --quiet pytest").ok:
                    return Holdout(0, total, "pytest is not available to run "
                                             "the held-out tests")
            report = f"{HOLDOUT_DIR}/junit.xml"
            run = box.run(
                f"PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 {python} -m pytest -q "
                f"-p no:cacheprovider --noconftest -c {HOLDOUT_DIR}/pytest.ini "
                f"--rootdir . --junitxml {report} "
                + " ".join(f"{HOLDOUT_DIR}/{p.name}" for p in files),
                timeout=600)
        junit = staging / report
        if not junit.is_file():
            return Holdout(0, total, "the held-out run wrote no report: "
                                     + run.summary(300))
        return Holdout(passed_tests(junit.read_text(errors="replace"), expected),
                       total)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


#: The ladder a finished workspace is graded on, weakest rung first. Each rung
#: is a strictly stronger claim about the deliverable than the one before it.
#: Eight small tasks scored pass/fail are eight bits, and at that N a real
#: improvement and a run of luck look alike; a factory that got two rungs
#: further on three tasks has visibly done better without anyone adding tasks
#: or spending more inference. Judged from a pristine export rather than from
#: the factory's own gates, for the reason :func:`score` is: what a release
#: would ship is the only thing worth grading.
LADDER = ("exports", "builds", "imports", "tests-present", "tests-pass",
          "benchmark")

#: Per-rung wall-clock ceiling. A project that hangs its own test suite should
#: lose that rung rather than stall the suite run queued behind it.
GRADE_TIMEOUT = 300

#: Directories whose Python is not the deliverable. Importing a test module
#: only reports the library's failure a second time, and ``build``/``dist``
#: hold copies of it that would be graded twice.
NOT_LIBRARY = frozenset({
    "tests", "test", "docs", "benchmarks", "experiments", "research",
    "artifacts", "examples", "scripts", "tickets", "build", "dist",
})

#: Module names that are scripts by convention. Importing one runs it, and a
#: benchmark's whole job is to take minutes, so a rung meant to ask whether
#: the library loads would instead be timing the sieve.
NOT_LIBRARY_MODULES = frozenset({
    "setup", "conftest", "main", "run", "bench", "benchmark", "demo", "example",
})

#: Written into the export, never into the workspace itself.
IMPORT_CHECK_FILE = "hermes_import_check.py"

#: Handed every top-level name at once, so the detail names each module that
#: broke rather than only the first. ``BaseException`` on purpose: a module
#: that parses argv at import time raises ``SystemExit``, and a library that
#: exits the interpreter on import is one that cannot be imported.
IMPORT_CHECK = """\
import importlib, sys
sys.path.insert(0, "src")
broken = 0
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except BaseException as exc:
        broken += 1
        print("%s: %s: %s" % (name, type(exc).__name__, exc))
sys.exit(1 if broken else 0)
"""


def importable(root: Path) -> list[str]:
    """Top-level names an export offers an importer, in either layout.

    Flat and ``src/`` are the two layouts the architect picks between and a
    package sits under only one of them, so both are looked at. Where there
    are packages they are the whole answer: importing a module runs it, and a
    loose ``bench.py`` beside the library is a script that exists to take
    minutes. Tests and ``setup.py`` are never the deliverable, and a name
    starting with an underscore is private or is ``__main__``, which runs the
    program on import.
    """
    packages: list[str] = []
    modules: list[str] = []
    for base in (Path(root), Path(root) / "src"):
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if entry.name.startswith((".", "_")) or entry.name in NOT_LIBRARY:
                continue
            if entry.is_dir() and (entry / "__init__.py").is_file():
                packages.append(entry.name)
            elif (entry.is_file() and entry.suffix == ".py"
                    and entry.stem not in NOT_LIBRARY_MODULES
                    and not entry.stem.startswith("test_")
                    and not entry.stem.endswith("_test")
                    and entry.stem.isidentifier()):
                modules.append(entry.stem)
    return sorted(set(packages or modules))


def test_functions(root: Path) -> int:
    """Test functions an export defines, counted from its source.

    From the source and not from a pytest run, because the rung below this one
    is whether the tests pass: a project that wrote no tests would otherwise
    report the same green suite as one that wrote them and passed, and writing
    none is the cheapest way for the factory to look finished.
    """
    total = 0
    for path in sorted(Path(root).rglob("*.py")):
        if not (path.name.startswith("test_") or path.stem.endswith("_test")):
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue  # a test file that does not parse defines no tests
        total += sum(1 for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith("test_"))
    return total


@dataclass
class Grade:
    """How far a finished workspace climbed :data:`LADDER`.

    A rung the project never declared -- no build command, no benchmark --
    is skipped and left out of the denominator, because an unasked question is
    not a failed one. ``exports`` is asked of every project, so a run that
    produced nothing scores 0 of 1 rather than the 0 of 0 that would read as
    unmeasured.
    """

    gates: list[Gate] = field(default_factory=list)
    #: Why the rungs are not a measurement of the project, when they are not.
    note: str = ""

    @property
    def passed(self) -> int:
        return sum(1 for g in self.gates if g.passed and not g.skipped)

    @property
    def total(self) -> int:
        return sum(1 for g in self.gates if not g.skipped)

    def mark(self, name: str, passed: bool, detail: str = "",
             skipped: bool = False) -> None:
        self.gates.append(Gate(name, passed, detail, skipped))

    def verdicts(self) -> dict[str, str]:
        """One word per rung, which is what the report and ``compare`` read."""
        return {g.name: ("skip" if g.skipped else "pass" if g.passed else "FAIL")
                for g in self.gates}


def grade(root: Path, sandbox_cfg: dict, commands: dict | None = None) -> Grade:
    """Climb the ladder against a pristine export of the workspace.

    Same export and same fresh sandbox as :func:`score`: a gate the factory
    ran for itself is the factory's own verdict, and the run that most needs
    grading is the one that crashed holding a workspace it never validated.
    No rung costs inference -- each is a command the project already declared
    or a question answered from its own source.
    """
    from .release import export
    from .sandbox import Sandbox

    commands = commands or {}
    result = Grade()
    root = Path(root)
    if not root.is_dir():
        result.mark("exports", False, "no workspace to grade")
        return result
    staging = Path(tempfile.mkdtemp(prefix="hermes-grade-", dir=root.parent))
    try:
        export(root, staging)
        sources = [p for p in staging.rglob("*")
                   if p.is_file() and p.suffix in SOURCE_SUFFIXES]
        result.mark("exports", bool(sources), f"{len(sources)} source file(s)")
        modules = importable(staging)
        cases = test_functions(staging)
        box = Sandbox(staging, dict(sandbox_cfg))
        # The host's interpreter when the box is the host, since a bare
        # ``python`` there may be one the project was never installed into.
        python = "python" if box.runtime == "docker" else shlex.quote(sys.executable)
        with box:
            # Setup is plumbing, not a rung: a project is not better for
            # needing dependencies, and the rungs after this one cannot run
            # without them.
            run_commands(box, [commands.get("setup", "")], "setup", Verification(),
                         required=False, install=True, autofix=False)
            _rung(box, result, "builds", commands.get("build", ""))
            if modules:
                box.write(IMPORT_CHECK_FILE, IMPORT_CHECK)
                run = box.run(f"{python} {IMPORT_CHECK_FILE} {' '.join(modules)}",
                              timeout=GRADE_TIMEOUT)
                result.mark("imports", run.ok,
                            ", ".join(modules) if run.ok else run.summary(400))
            elif any(p.suffix == ".py" for p in sources):
                result.mark("imports", False, "no top-level module to import")
            else:
                result.mark("imports", False, "not a Python project", skipped=True)
            result.mark("tests-present", cases > 0, f"{cases} test function(s)")
            _rung(box, result, "tests-pass", commands.get("test", ""))
            _rung(box, result, "benchmark", commands.get("benchmark", ""))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return result


def _rung(box, result: Grade, name: str, command: str) -> None:
    """Run one declared command and fold its single gate into the ladder.

    Through :func:`~hermes.verify.run_commands` rather than around it, so a
    rung means here exactly what the same command meant when the factory ran
    it -- including the fallback ``true`` counting as no command at all, which
    is what stops an architect that skipped the question from being credited
    with an answer.
    """
    scratch = Verification()
    run_commands(box, [command], name, scratch, timeout=GRADE_TIMEOUT)
    result.gates.extend(scratch.gates)


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
    #: Iterations the loop got through, which is what separates a task that
    #: stalled from one that failed fast.
    iterations: int = 0
    #: True when the wall-clock budget stopped the run, rather than the
    #: factory reaching a terminal state of its own accord.
    timed_out: bool = False
    #: Wall clock neither ledger explains, and wall clock both claim.
    #: Prefetch generates while gates run, so the ledgers may legitimately
    #: exceed the clock. Both numbers are reported instead of assumed to be
    #: zero, because a split nobody checked is a split nobody should trust.
    unaccounted_seconds: float = 0.0
    overlap_seconds: float = 0.0
    error: str = ""
    root: str = ""
    #: Held-out tests passed, of those the task's suite declares. Zero of
    #: zero for a task with no held-out suite.
    holdout_passed: int = 0
    holdout_total: int = 0
    holdout_note: str = ""
    #: Rungs of :data:`LADDER` the finished workspace passed, of those it was
    #: asked. Dense where ``completed`` is one bit: two runs that both failed
    #: every task can still be several rungs apart.
    grade_passed: int = 0
    grade_total: int = 0
    #: Rung name -> pass/FAIL/skip, so the suite can say which rung the factory
    #: loses its projects on rather than only that the number moved.
    grade_gates: dict[str, str] = field(default_factory=dict)
    grade_note: str = ""

    def row(self) -> str:
        mark = "pass" if self.completed else ("TIME" if self.timed_out else "FAIL")
        return (f"{self.name:<14} {mark:>4} {self.phase:<14} "
                f"tickets={self.tickets_completed}/{self.tickets} "
                f"first_pass={self.first_pass:.0%} "
                + (f"rungs={self.grade_passed}/{self.grade_total} "
                   if self.grade_total else "")
                + (f"held-out={self.holdout_passed}/{self.holdout_total} "
                   if self.holdout_total else "")
                + f"{self.seconds / 60:.1f}m "
                f"(gpu {self.inference_seconds / 60:.1f}m, "
                f"cpu {self.sandbox_seconds / 60:.1f}m)"
                + (f" error={self.error[:60]}" if self.error else ""))


def collect(name: str, state, board, usage: list[dict], sandbox_seconds: float,
            seconds: float, error: str = "", root: str = "",
            timed_out: bool = False, holdout: Holdout | None = None,
            graded: Grade | None = None) -> Trial:
    """Turn a finished run into numbers.

    Kept free of the factory so the accounting can be tested without standing
    up a project or calling a model. Everything is read defensively: the runs
    worth measuring include the ones that crashed or ran out of clock, and a
    partial run has written less than a finished one.
    """
    counts = board.counts()
    done = board.by_status("completed")
    inference = round(sum(c.get("seconds", 0.0) for c in usage), 1)
    ledgers = inference + sandbox_seconds
    holdout = holdout or Holdout()
    graded = graded or Grade()
    return Trial(
        name=name,
        phase=getattr(state, "phase", "UNKNOWN"),
        completed=getattr(state, "phase", "") == "COMPLETED",
        tickets=len(board.tickets),
        tickets_completed=counts.get("completed", 0),
        tickets_abandoned=counts.get("abandoned", 0),
        first_pass=(sum(1 for t in done if t.attempts <= 1) / len(done)) if done else 0.0,
        seconds=round(seconds, 1),
        inference_seconds=inference,
        sandbox_seconds=round(sandbox_seconds, 1),
        model_calls=len(usage),
        chars_out=sum(c.get("chars_out", 0) for c in usage),
        iterations=int(getattr(state, "iteration", 0) or 0),
        timed_out=timed_out,
        unaccounted_seconds=round(max(0.0, seconds - ledgers), 1),
        overlap_seconds=round(max(0.0, ledgers - seconds), 1),
        error=error,
        root=root,
        holdout_passed=holdout.passed,
        holdout_total=holdout.total,
        holdout_note=holdout.note,
        grade_passed=graded.passed,
        grade_total=graded.total,
        grade_gates=graded.verdicts(),
        grade_note=graded.note,
    )


def run_task(task: Task, root: Path, config: Config | None = None,
             max_iterations: int | None = None, quiet: bool = True,
             budget: float | None = None) -> Trial:
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
        factory.run(max_iterations=max_iterations or task.iterations,
                    deadline=started + (budget or task.budget_seconds))
    except Exception as exc:  # noqa: BLE001 - a crashed entry is a result, not a stop
        error = f"{type(exc).__name__}: {exc}"[:300]
    seconds = time.time() - started  # the scoring is ours, not the factory's
    # Whatever state the run ended in is what gets measured: a crashed or
    # timed-out run is exactly the one whose partial project is worth grading.
    config = factory.config if factory is not None else (config or Config.load())
    commands = ((factory.state.plan.get("commands") or {})
                if factory is not None else {})
    sandbox_cfg = config.data.get("sandbox", {})
    try:
        graded = grade(root, sandbox_cfg, commands)
    except Exception as exc:  # noqa: BLE001 - a grading fault is reported, not raised
        graded = Grade(note=f"grading failed: {type(exc).__name__}: {exc}"[:300])
    holdout = None
    if task.holdout:
        try:
            holdout = score(task, root, sandbox_cfg, commands.get("setup", ""))
        except Exception as exc:  # noqa: BLE001 - a scoring fault is reported, not raised
            holdout = Holdout(0, sum(expected_tests(SUITE_DIR / task.holdout).values()),
                              f"scoring failed: {type(exc).__name__}: {exc}"[:300])
    if factory is None:
        return collect(task.name, State(idea=task.idea), Board(root), [], 0.0,
                       seconds, error, str(root), holdout=holdout, graded=graded)
    return collect(task.name, factory.state, factory.board,
                   list(getattr(factory.router, "usage", [])),
                   float(getattr(getattr(factory, "sandbox", None), "spent", 0.0)),
                   seconds, error, str(root),
                   timed_out=bool(factory.state.metrics.get("deadline_reached")),
                   holdout=holdout, graded=graded)


def run_suite(root: Path, config: Config | None = None,
              names: list[str] | None = None, max_iterations: int | None = None,
              quiet: bool = True, on_trial=None,
              budget: float | None = None) -> list[Trial]:
    """Run the suite serially.

    Serially on purpose: two projects sharing one GPU measure each other's
    queueing delay rather than the factory, and the wall-clock number is half
    the point of running this at all.
    """
    root = Path(root)
    chosen = [t for t in SUITE if not names or t.name in names]
    trials = []
    for task in chosen:
        trial = run_task(task, root / task.name, config, max_iterations, quiet,
                         budget)
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
    timed = sum(1 for t in trials if t.timed_out)
    lines = [t.row() for t in trials] + [
        "",
        f"completed      {sum(1 for t in trials if t.completed)}/{len(trials)}",
        f"first-pass     {(first / weight if weight else 0):.0%} of "
        f"{weight} completed tickets",
        f"wall clock     {seconds / 60:.1f}m",
        f"inference      {inference / 60:.1f}m ({inference / seconds:.0%})",
        f"sandbox        {sandbox / 60:.1f}m ({sandbox / seconds:.0%})",
        f"unaccounted    {sum(t.unaccounted_seconds for t in trials) / 60:.1f}m "
        f"of clock neither ledger explains",
        f"overlap        {sum(t.overlap_seconds for t in trials) / 60:.1f}m "
        f"claimed by both ledgers, which prefetch makes possible",
        ladder(trials),
        resolution(trials),
    ]
    ungraded = [t.name for t in trials if t.grade_note]
    if ungraded:
        lines.append(f"ungraded       {', '.join(ungraded)}: the ladder did not "
                     "measure the project (see grade_note)")
    scored = [t for t in trials if t.holdout_total]
    if scored:
        passed = sum(t.holdout_passed for t in scored)
        total = sum(t.holdout_total for t in scored)
        lines.append(f"held-out       {passed / total:.0%} of {total} tests across "
                     f"{len(scored)} library task(s)")
        # The failure NL2Repo-Bench names first: stopping early and calling
        # a half-built library done. Only tests the factory never saw show it.
        claimed = [t.name for t in scored
                   if t.completed and t.holdout_passed < t.holdout_total]
        if claimed:
            lines.append(f"overclaimed    {len(claimed)} task(s) reached COMPLETED "
                         f"while failing held-out tests: {', '.join(claimed)}")
        unmeasured = [t.name for t in scored if t.holdout_note]
        if unmeasured:
            lines.append(f"unscored       {', '.join(unmeasured)}: the held-out "
                         "run did not measure the project (see holdout_note)")
    if timed:
        lines.append(f"timed out      {timed} task(s) stopped on their budget, "
                     "so their numbers are a floor rather than a result")
    if not any(t.model_calls for t in trials):
        lines.append("warning        no model calls recorded: these numbers "
                     "describe a stub, not a model")
    return "\n".join(lines)


def ladder(trials: list[Trial]) -> str:
    """Where on the ladder the suite is losing its projects.

    The rate says a change helped; the per-rung line says what it helped at,
    which is the difference between a number and somewhere to work next. A
    rung a project was never asked -- no benchmark declared, say -- is absent
    from both, so the denominators differ per rung on purpose.
    """
    graded = [t for t in trials if t.grade_total]
    if not graded:
        return "ladder         no workspace was graded"
    rungs = []
    for name in LADDER:
        asked = [t for t in graded if t.grade_gates.get(name, "skip") != "skip"]
        if asked:
            ok = sum(1 for t in asked if t.grade_gates.get(name) == "pass")
            rungs.append(f"{name} {ok}/{len(asked)}")
    passed = sum(t.grade_passed for t in graded)
    total = sum(t.grade_total for t in graded)
    return (f"ladder         {passed / total:.0%} of {total} rung(s) asked "
            f"across {len(graded)} task(s)\n"
            + " " * 15 + "  ".join(rungs))


def resolution(trials: list[Trial]) -> str:
    """Whether this suite can rank a change at all.

    A suite where every task passes, or every task fails, produces real
    numbers that cannot separate a better factory from a worse one. Saying so
    is the difference between a measurement and a decoration, and it is the
    check that catches a suite whose tasks are too easy or too hard for the
    model in front of it.
    """
    if len(trials) < 2:
        return "resolution     too few tasks to rank anything"
    completed = sum(1 for t in trials if t.completed)
    rates = {round(t.first_pass, 2) for t in trials if t.tickets_completed}
    # Partial credit is what lets a suite nobody finishes still rank: two
    # factories that both fail can differ in how much of the library works.
    held = {round(t.holdout_passed / t.holdout_total, 2)
            for t in trials if t.holdout_total}
    # The ladder is what separates two runs that both finished nothing: a
    # suite stuck at `imports` and one stuck at `tests-pass` are different
    # factories, and `completed` cannot say so.
    rungs = {round(t.grade_passed / t.grade_total, 2)
             for t in trials if t.grade_total}
    if completed == 0 and len(held) <= 1 and len(rungs) <= 1:
        return ("resolution     NONE: no task completed. Nothing here can rank "
                "a change; fix the factory or the model first")
    if (completed == len(trials) and len(rates) <= 1 and len(held) <= 1
            and len(rungs) <= 1):
        return ("resolution     NONE: every task completed at one first-pass "
                "rate, so this suite cannot tell two factories apart")
    return (f"resolution     {completed}/{len(trials)} completed, "
            f"{len(rates)} distinct first-pass rate(s)"
            + (f", {len(held)} distinct held-out rate(s)" if held else "")
            + (f", {len(rungs)} distinct ladder rate(s)" if rungs else ""))


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
        held = ("  held-out {} -> {}".format(a.get("holdout_passed", 0),
                                           b.get("holdout_passed", 0))
                if a.get("holdout_total") or b.get("holdout_total") else "")
        rungs = ("  rungs {}/{} -> {}/{}".format(
            a.get("grade_passed", 0), a.get("grade_total", 0),
            b.get("grade_passed", 0), b.get("grade_total", 0))
            if a.get("grade_total") or b.get("grade_total") else "")
        lines.append(f"{name:<14} {done:>18} {passed:>16} {clock:>18}{held}{rungs}")
    return "\n".join(lines + ["", _verdict(old, new)])


def _verdict(old: dict, new: dict) -> str:
    """What this pair of runs does and does not support concluding."""
    shared = set(old) & set(new)
    if not shared:
        return "nothing to compare: the two runs share no task"
    if not any(t.get("model_calls") for t in list(old.values()) + list(new.values())):
        return ("neither run made a model call: this compares two stubs and "
                "cannot justify a change")
    moved = sorted(
        name for name in shared
        if old[name]["completed"] != new[name]["completed"]
        or abs(old[name].get("first_pass", 0) - new[name].get("first_pass", 0)) > 0.01
        or old[name].get("holdout_passed", 0) != new[name].get("holdout_passed", 0)
        or (old[name].get("grade_passed", 0), old[name].get("grade_total", 0))
        != (new[name].get("grade_passed", 0), new[name].get("grade_total", 0))
    )
    if not moved:
        return ("no task changed completion, first-pass rate, held-out tests "
                "or ladder rungs: this pair does not support accepting or "
                "rejecting the change")
    return f"{len(moved)} of {len(shared)} task(s) moved: {', '.join(moved)}"
