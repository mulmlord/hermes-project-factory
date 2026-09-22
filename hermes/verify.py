"""Verification: build, tests, quality gates, review and failure triage."""
from __future__ import annotations

import ast
import json
import os
import random
import re
import shlex
import time
from dataclasses import dataclass, field
from functools import partial
from itertools import count, zip_longest
from pathlib import Path

from . import prompts
from .agents import Prompt
from .config import Config
from .log import Logger, NULL
from .sandbox import Result, Sandbox, SandboxError
from .tickets import split_validation
from .tools import Lookup, Toolbox

#: Patterns that mark generated code as unsafe regardless of test results.
UNSAFE_PATTERNS = [
    (r"\brm\s+-rf\s+/(?:\s|$|\*)", "recursive delete of the filesystem root"),
    (r"\bcurl\b[^\n|]*\|\s*(?:ba)?sh", "piping a download straight into a shell"),
    (r"\bwget\b[^\n|]*\|\s*(?:ba)?sh", "piping a download straight into a shell"),
    (r"subprocess\.[A-Za-z_]+\([^)]*shell\s*=\s*True[^)]*\+", "shell=True with concatenated input"),
    (r"\beval\s*\(\s*(?:input|request|sys\.argv)", "eval on untrusted input"),
    (r"\bos\.system\s*\([^)]*[+%]", "os.system with interpolated input"),
    (r"~/\.ssh|/\.aws/credentials|id_rsa\b", "reading host credentials"),
    (r"(?i)(api[_-]?key|secret|password)\s*=\s*['\"][A-Za-z0-9/+_-]{16,}", "hardcoded secret"),
]

SOURCE_SUFFIXES = {".py", ".js", ".ts", ".sh", ".rb", ".go", ".rs", ".c", ".cc",
                   ".cpp", ".h", ".hpp", ".java", ".yaml", ".yml", ".toml"}


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False

    def __str__(self) -> str:
        mark = "skip" if self.skipped else ("pass" if self.passed else "FAIL")
        return f"[{mark}] {self.name}" + (f": {self.detail[:200]}" if self.detail else "")


@dataclass
class Verification:
    gates: list[Gate] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed or g.skipped for g in self.gates)

    @property
    def failures(self) -> list[Gate]:
        return [g for g in self.gates if not g.passed and not g.skipped]

    def report(self) -> str:
        return "\n".join(str(g) for g in self.gates)

    def failure_output(self, limit: int = 6000) -> str:
        """The tail of what failed: a build log puts the error last."""
        if limit <= 0:            # [-0:] is the whole string, which is not "none"
            return ""
        chunks = [r.summary() for r in self.results if not r.ok]
        return ("\n\n".join(chunks) or self.report())[-limit:]


_MISSING_TOOL = re.compile(r"([\w.+-]+): (?:not found|command not found)")

#: Tools a gate may legitimately need. Anything else named by a model-written
#: command is not fetched from PyPI: that would make typosquatting a supply
#: chain into the sandbox.
DEV_TOOLS = frozenset({
    "pytest", "pytest-cov", "flake8", "ruff", "mypy", "black", "pylint",
    "isort", "coverage", "tox", "hypothesis", "bandit",
})


def _install_missing(sandbox: Sandbox, result, verification: 'Verification',
                     log: Logger) -> bool:
    """Exit 127 means a tool is absent, not that the generated code is wrong.

    The architect is meant to install its tooling in the setup command and
    routinely forgets, which otherwise fails every gate for the rest of the run.
    """
    found = _MISSING_TOOL.search(result.stderr + result.stdout)
    if found:
        tool = found.group(1)
    elif result.exit_code == 127:
        # A pipeline hides 127 behind the exit code of its last stage, so the
        # message above is the reliable signal and this is only the fallback.
        tool = (shlex.split(result.command or "")[:1] or [""])[0]
    else:
        return False
    if tool in ("", "sh", "bash", "python", "python3"):
        return False
    if sandbox.runtime != "docker":
        return False  # installing onto the host is not ours to do
    if tool in DEV_TOOLS:
        target = tool
    elif any((sandbox.root / f).is_file() for f in ("pyproject.toml", "setup.py")):
        target = "-e ."  # a missing console script means the project is uninstalled
    else:
        return False
    log.info("verify.install_missing", tool=tool, install=target)
    attempt = sandbox.install(f"pip install --quiet {target}")
    if not attempt.ok:
        # Why the install failed is the actual diagnosis (a src/ layout with no
        # package_dir, say), so it has to reach the agent that must fix it.
        verification.results.append(attempt)
        return False
    sandbox.installed.append(target)
    return True


def run_commands(sandbox: Sandbox, commands: list[str], name: str,
                 verification: Verification, *, required: bool = True,
                 timeout: int | None = None, install: bool = False,
                 autofix: bool = True, log: Logger = NULL) -> None:
    """Run a command list and fold the outcome into *verification* as one gate."""
    commands = [c for c in commands if c and c.strip() and c.strip() != "true"]
    if not commands:
        verification.gates.append(Gate(name, True, "no command configured", skipped=True))
        return
    for command in commands:
        runner = sandbox.install if install else sandbox.run
        result = runner(command, timeout=timeout)
        if not result.ok and autofix and _install_missing(sandbox, result, verification, log):
            result = runner(command, timeout=timeout)
        verification.results.append(result)
        if not result.ok:
            verification.gates.append(
                Gate(name, not required, result.summary(1500), skipped=not required))
            log.warn("verify.failed", gate=name, command=command[:80],
                     exit=result.exit_code)
            return
    verification.gates.append(Gate(name, True, f"{len(commands)} command(s) ok"))


def _sources(root: Path, paths: list[str] | None = None) -> dict[str, str]:
    """Generated sources and their text, keyed by the path a hit will name.

    Shared by both scanners so they can never disagree about what counts as
    project source: a classifier that walked a different tree to the regexes
    would report findings against files the gate had already excused.
    """
    root = Path(root)
    targets = ([root / p for p in paths] if paths else
               [p for p in root.rglob("*")
                if p.suffix in SOURCE_SUFFIXES
                and not any(part in (".git", ".hermes", "node_modules", ".venv",
                                     "artifacts", "__pycache__")
                            for part in p.parts)])
    found: dict[str, str] = {}
    for path in targets:
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        rel = path.relative_to(root) if root in path.parents else path
        found[str(rel)] = text
    return found


def scan_unsafe(root: Path, paths: list[str] | None = None) -> list[str]:
    """Static security scan over generated sources. Returns human-readable hits."""
    return _patterns(_sources(root, paths))


def _patterns(sources: dict[str, str]) -> list[str]:
    """Every pattern hit in ``{where: text}``, named by where and by line."""
    hits = []
    for rel, text in sources.items():
        for pattern, description in UNSAFE_PATTERNS:
            for match in re.finditer(pattern, text):
                line = text[: match.start()].count("\n") + 1
                hits.append(f"{rel}:{line}: {description}")
    return hits


#: Command shapes that add software to the sandbox, and what to call the thing
#: doing it. Over-matching is the safe direction here: the cost of a false hit
#: is one refused proposal, and the cost of a miss is a dependency chosen by
#: whoever wrote a web page.
INSTALLERS = [
    (r"\b(?:pip|pip3|pipx|uv|conda|mamba|poetry|pdm)\b[^\n;&|]*?\b(?:install|add)\b",
     "a python package manager"),
    (r"\bpython3?\s+-m\s+pip\b[^\n;&|]*?\binstall\b", "pip"),
    (r"\b(?:npm|pnpm|yarn|bun)\b[^\n;&|]*?\b(?:install|add|i)\b",
     "a node package manager"),
    (r"\b(?:cargo|gem|go|dotnet|composer|nix-env)\b[^\n;&|]*?\b(?:install|add|get)\b",
     "a language package manager"),
    (r"\b(?:apt|apt-get|dnf|yum|apk|brew|zypper)\b[^\n;&|]*?\b(?:install|add)\b",
     "a system package manager"),
    (r"\bpacman\s+-[A-Za-z]*S", "pacman"),
    (r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:ba)?sh", "a piped installer"),
]


def installs(command: str) -> str:
    """Name what this command would install with, or ``""`` if it installs nothing.

    Kept beside the security patterns because it answers the same kind of
    question about a string, but it is not a security verdict: ``pip install
    pytest`` is what the architect is supposed to write in the setup command.
    It is only a refusal when the caller has decided that *this* source of
    commands does not get to add packages.
    """
    return next((manager for pattern, manager in INSTALLERS
                 if re.search(pattern, command, re.IGNORECASE)), "")


def security_gate(root: Path, paths: list[str] | None = None, *,
                  classifier=None, clean: str = "no unsafe patterns") -> Gate:
    """The security gate: patterns that block, plus findings that only warn.

    The pattern list keeps its veto. A classifier is a second opinion on the
    much larger set of defects no regex expresses, and by default it can only
    add a line to the detail -- which the reviewer and the triager both read,
    because both are shown the gate report. That is the point: the cheap model
    says where to look and the expensive one decides whether it was right.

    ``security.classifier.block`` promotes findings to blocking, for an
    operator who has tuned a checkpoint against their own code and knows its
    false positive rate. Until then a guessed defect must not fail a ticket
    that was correct.
    """
    return _gate(_sources(root, paths), classifier=classifier, clean=clean)


def text_gate(texts: dict[str, str], *, classifier=None, block: bool = False,
              clean: str = "clean") -> Gate:
    """The same gate, over text that is not a file on disk.

    What gets screened here is a string the factory is about to act on but
    has not written down anywhere -- a shell command proposed by a research
    round, in practice. Reusing :func:`scan_unsafe`'s own pattern list and
    the same classifier call is deliberate: two security checks that
    disagreed about what ``curl … | sh`` is would be worse than one.

    *block* promotes the classifier's findings from advisory to blocking.
    Callers screening untrusted material set it, and the asymmetry with
    :func:`security_gate` is the point: a false positive there fails a ticket
    the factory itself wrote and costs a whole generate/verify cycle, while a
    false positive here refuses one command a fetched page asked for.
    """
    return _gate({k: v for k, v in texts.items() if str(v).strip()},
                 classifier=classifier, clean=clean, block=block)


def _gate(sources: dict[str, str], *, classifier=None, clean: str,
          block: bool = False) -> Gate:
    """One security verdict over ``{where: text}``, whatever it was read from."""
    hits = _patterns(sources)
    findings = classifier.scan(sources) if classifier is not None else []
    advisories = [f.render() for f in findings]
    blocking = hits + (advisories if findings and
                       (block or classifier.cfg.block) else [])
    detail = " | ".join(part for part in (
        "; ".join(hits[:5]),
        ("review: " + "; ".join(advisories[:5])) if advisories else "",
    ) if part) or clean
    return Gate("security", not blocking, detail)


def record_installs(sandbox: Sandbox, plan: dict) -> None:
    """Fold anything Hermes had to install into the project's own setup command.

    Otherwise the clean-room check would fail on a dependency that only exists
    because Hermes quietly installed it, and the published repo would not build.
    """
    commands = plan.setdefault("commands", {})
    setup = (commands.get("setup") or "").strip()
    pending = [t for t in sandbox.installed if t not in setup]
    if not pending:
        return
    added = "pip install " + " ".join(pending)
    commands["setup"] = added if setup in ("", "true") else f"{setup} && {added}"


#: Operators the mutation gate applies, one change per mutant. The set is
#: small on purpose. Every entry here alters behaviour a test that asserts on
#: results will see, and the operators whose mutants are routinely *equivalent*
#: to the original -- deleting a statement, perturbing a string, dropping a
#: call -- are left out. A survivor nobody could ever kill spends a repair
#: cycle and teaches the factory nothing.
_COMPARE = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
            ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
            ast.Is: ast.IsNot, ast.IsNot: ast.Is,
            ast.In: ast.NotIn, ast.NotIn: ast.In}
_BINARY = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
_BOOLEAN = {ast.And: ast.Or, ast.Or: ast.And}

_SYMBOLS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==",
            ast.NotEq: "!=", ast.Is: "is", ast.IsNot: "is not", ast.In: "in",
            ast.NotIn: "not in", ast.Add: "+", ast.Sub: "-", ast.Mult: "*",
            ast.Div: "/", ast.And: "and", ast.Or: "or"}


@dataclass
class MutationBudget:
    """What one mutation gate may spend, and how hard it may push.

    Each mutant is a whole run of the project's test command, so the wall
    clock and not the mutant count is what actually bounds this: against a
    suite that takes a minute the default budget buys two mutants, which is
    the honest answer for a suite that takes a minute.
    """
    seconds: float = 120.0
    mutants: int = 20
    timeout: int = 120

    @classmethod
    def from_config(cls, config: Config) -> "MutationBudget | None":
        """Build one from ``verification.mutation``, or None when disabled."""
        spec = dict(config.get("verification.mutation") or {})
        if not spec.pop("enabled", False):
            return None
        return cls(**{k: v for k, v in spec.items()
                      if k in cls.__dataclass_fields__})


def _new_constant(value):
    """A different constant of the same kind, or None where changing it is noise.

    Strings are left alone. A changed message is a mutant only a test that
    asserts on prose can kill, so the survivor it leaves says nothing about
    whether the behaviour around it is tested -- which is the one question
    this gate exists to ask.

    A float past 2**53 absorbs the increment: ``1e16 + 1 == 1e16``. Refusing
    it here is what keeps every operator's promise that a mutant differs from
    its original, and a mutant identical to the code it replaced is the one
    result this gate must never produce -- the tests pass, and it reports a
    survivor where nothing was ever changed.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)) and value + 1 != value:
        return value + 1
    return None


def _mutations(tree: ast.AST):
    """Every single-point mutation of *tree*, in a stable walk order.

    Yields ``(line, description, apply)``, where *apply* makes that one change
    to the tree it was walked from. Callers re-parse between mutants rather
    than undoing: an AST edited in place and edited back is one bug away from
    a mutant that stays quietly applied to every run after it.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for index, op in enumerate(node.ops):
                if (swap := _COMPARE.get(type(op))) is not None:
                    yield (node.lineno, f"{_SYMBOLS[type(op)]} -> {_SYMBOLS[swap]}",
                           partial(node.ops.__setitem__, index, swap()))
        elif isinstance(node, ast.BinOp):
            if (swap := _BINARY.get(type(node.op))) is not None:
                yield (node.lineno,
                       f"{_SYMBOLS[type(node.op)]} -> {_SYMBOLS[swap]}",
                       partial(setattr, node, "op", swap()))
        elif isinstance(node, ast.BoolOp):
            swap = _BOOLEAN[type(node.op)]
            yield (node.lineno, f"{_SYMBOLS[type(node.op)]} -> {_SYMBOLS[swap]}",
                   partial(setattr, node, "op", swap()))
        elif isinstance(node, ast.Constant):
            if (value := _new_constant(node.value)) is not None:
                yield (node.lineno, f"{node.value!r} -> {value!r}",
                       partial(setattr, node, "value", value))


def _mutable(source: str) -> int:
    """How many mutants *source* has, or 0 when it must not be mutated at all.

    A file is refused when ``ast.unparse`` does not reproduce its own tree.
    The gate rewrites the whole file from the mutated AST, so an unparse that
    is not faithful would score the mutant against a second change nobody
    asked for -- and the verdict would be about that change instead.
    """
    try:
        tree = ast.parse(source)
        if ast.dump(ast.parse(ast.unparse(tree))) != ast.dump(tree):
            return 0
    except (SyntaxError, ValueError, RecursionError):
        return 0
    return sum(1 for _ in _mutations(tree))


def _mutant(source: str, index: int) -> tuple[str, str, int] | None:
    """The *index*-th mutant of *source*, as ``(code, description, line)``."""
    tree = ast.parse(source)
    for position, (line, description, apply) in enumerate(_mutations(tree)):
        if position == index:
            apply()
            return ast.unparse(tree), description, line
    return None


def _mutation_targets(sandbox: Sandbox, changed: list[str]) -> dict[str, int]:
    """The ticket's own mutable Python sources, and how many mutants each has.

    Test files are excluded as targets. Mutating one measures nothing: the
    change breaks the assertion that contains it, which is indistinguishable
    from the suite catching a defect. Only the code under test is worth
    changing.
    """
    found: dict[str, int] = {}
    for path in dict.fromkeys(changed):
        parts = Path(path).parts
        name = Path(path).name
        if (not name.endswith(".py") or name.startswith("test_")
                or name.endswith("_test.py") or name == "conftest.py"
                or any(part in ("test", "tests") for part in parts)):
            continue
        try:
            source = sandbox.read(path)
        except SandboxError:
            continue
        if source and (count := _mutable(source)):
            found[path] = count
    return found


def _place(sandbox: Sandbox, path: str, code: str, stamp: float) -> None:
    """Write a source back, and make sure the next reader recompiles it.

    CPython validates a ``__pycache__`` entry against its source's size and
    its mtime in whole seconds. Two same-length mutants written inside one
    second are indistinguishable from the original to that check, so the
    stale bytecode runs, the mutant scores as a survivor it never was, and
    the gate quietly stops measuring the thing it was turned on for. Giving
    every write its own mtime is what makes the measurement real.
    """
    os.utime(sandbox.write(path, code), (stamp, stamp))


def _schedule(targets: dict[str, int], limit: int) -> list[tuple[str, int]]:
    """Which mutants to try, drawn evenly from each file, in a fixed order.

    Sampling beats taking the first *limit* sites in walk order, which would
    only ever mutate the outermost statements of the longest file. The seed is
    fixed so two verifications of the same tree ask the same question: a gate
    whose verdict moved between attempts would have the repair loop chasing
    noise instead of the weakness it reported.
    """
    rng = random.Random(0)
    rows = []
    for path, count in sorted(targets.items()):
        indices = list(range(count))
        rng.shuffle(indices)
        rows.append([(path, index) for index in indices])
    return [pair for row in zip_longest(*rows) for pair in row if pair][:limit]


def mutation_gate(sandbox: Sandbox, plan: dict, changed: list[str],
                  budget: MutationBudget | None, *, tests_passed: bool = False,
                  log: Logger = NULL) -> Gate:
    """Change the code under the tests and see whether the tests notice.

    Coverage says a line ran. It cannot say an assertion depended on it, and
    here the same system writes the code and the tests, so a suite that only
    executes what it was written beside is the expected outcome rather than an
    unlucky one. Mutation testing asks the question directly: flip a
    comparison, and if every test still passes, nothing was checking.

    The gate is scoped to the files the ticket wrote and bounded by a wall
    clock, because each mutant costs a full run of the project's test command.
    A survivor becomes the gate's detail, which the triager and the reviewer
    both read, and that is the point of running it at all: "the tests pass
    with ``>=`` for ``>`` at fee.py:2" is a repair instruction, where "add
    more tests" is a wish.

    It runs only once the tests have actually passed. Against a red suite
    every mutant fails for the reason the suite was already failing, so the
    gate would report a perfect score over a measurement it never made.

    Scoping to *changed* has a consequence worth knowing: a ticket that writes
    only tests has no source of its own to mutate, and a ticket that writes
    source before any suite exists has no tests to run. So the gate weighs a
    suite against code that arrived with it, or against a later change to code
    an existing suite already covers, and stays quiet either side of that.
    Widening it to whatever the tests might reach would need coverage the
    factory does not collect, and would price the gate out of a ticket.

    The known false positive is the equivalent mutant: a change that cannot
    alter behaviour, which no test can kill. :data:`_COMPARE` and the rest are
    chosen to make those uncommon rather than impossible, and
    ``execution.max_ticket_attempts`` bounds what one costs. That residual is
    why ``verification.mutation`` is off until an operator asks for it.

    While it runs, the ticket's sources are transiently the mutant instead of
    what the agent wrote. Nothing reads them but the test command -- except a
    prefetch generating the next ticket against the same workspace, which may
    catch one flipped operator and a file with its comments stripped. They go
    back either way, and a prefetched candidate is a first guess the gates
    still judge.
    """
    if budget is None:
        return Gate("mutation", True, "not enabled", skipped=True)
    if not tests_passed:
        return Gate("mutation", True, "the tests did not pass", skipped=True)
    command = ((plan.get("commands") or {}).get("test") or "").strip()
    if not command or command == "true":
        return Gate("mutation", True, "no test command", skipped=True)
    targets = _mutation_targets(sandbox, changed)
    if not targets:
        return Gate("mutation", True, "no mutable python source changed", skipped=True)

    original = {path: sandbox.read(path) for path in targets}
    stamps = {path: sandbox.resolve(path).stat().st_mtime for path in targets}
    deadline = time.time() + float(budget.seconds)
    stamp = count(time.time() + 1.0)

    def restore() -> None:
        """Put the sources back exactly as they were found, mtimes included.

        A mutant left behind would replace the agent's file with unparsed
        source -- every comment in it gone -- so this has to survive whatever
        the sandbox raises, not only the paths that return normally.
        """
        for path, text in original.items():
            _place(sandbox, path, text, stamps[path])

    # Every mutant is written back through ast.unparse, which drops comments
    # and normalises formatting. A test command that reacts to either -- a
    # lint step in front of the suite being the usual one -- would fail on all
    # of them and leave the gate reporting a flawless score it never measured.
    # So the control is the unparsed tree with nothing else changed: if that
    # does not pass, this gate cannot measure anything here and has to say so.
    try:
        for path, text in original.items():
            _place(sandbox, path, ast.unparse(ast.parse(text)), next(stamp))
        control = sandbox.run(command, timeout=int(budget.timeout))
    finally:
        restore()
    if not control.ok:
        log.info("verify.mutation_uncontrolled", files=len(targets))
        return Gate("mutation", True, "the tests do not pass on unparsed source",
                    skipped=True)

    survivors: list[str] = []
    tried = 0
    try:
        for path, index in _schedule(targets, int(budget.mutants)):
            if time.time() >= deadline:
                break
            made = _mutant(original[path], index)
            if made is None:
                continue
            code, description, line = made
            _place(sandbox, path, code, next(stamp))
            result = sandbox.run(command, timeout=int(budget.timeout))
            _place(sandbox, path, original[path], stamps[path])
            tried += 1
            if result.ok:
                survivors.append(f"{path}:{line}: {description}")
                log.info("verify.mutant_survived", path=path, line=line,
                         change=description)
    finally:
        restore()

    log.info("verify.mutation", tried=tried, survived=len(survivors),
             files=len(targets))
    if not tried:
        return Gate("mutation", True, "no mutant fitted the budget", skipped=True)
    if survivors:
        return Gate("mutation", False,
                    f"{len(survivors)}/{tried} mutants survived; the tests pass "
                    f"unchanged with each of these: {'; '.join(survivors[:2])}")
    return Gate("mutation", True, f"{tried} mutants, all killed")


def verify_project(sandbox: Sandbox, plan: dict, *, include_benchmark: bool = False,
                   classifier=None, log: Logger = NULL) -> Verification:
    """The full project gate: setup, build, lint, tests, security."""
    commands = plan.get("commands") or {}
    verification = Verification()
    run_commands(sandbox, [commands.get("setup", "")], "setup", verification,
                 required=False, install=True, log=log)
    run_commands(sandbox, [commands.get("build", "")], "build", verification, log=log)
    run_commands(sandbox, [commands.get("lint", "")], "lint", verification,
                 required=False, log=log)
    run_commands(sandbox, [commands.get("test", "")], "tests", verification, log=log)
    if include_benchmark and commands.get("benchmark"):
        run_commands(sandbox, [commands["benchmark"]], "benchmark", verification,
                     required=False, log=log)
    verification.gates.append(security_gate(sandbox.root, classifier=classifier))
    record_installs(sandbox, plan)
    return verification


def verify_ticket(sandbox: Sandbox, ticket, plan: dict, changed: list[str],
                  log: Logger = NULL, tests_required: bool = True,
                  classifier=None,
                  mutation: MutationBudget | None = None) -> Verification:
    """Gate a single ticket: its own validation commands plus the project tests.

    Early in a project the test suite does not exist yet, so failing it would
    block the very ticket that creates it. Until the suite has passed once,
    it is advisory; from then on it is a hard regression gate.

    *mutation* asks the further question of a suite that did pass: whether it
    would have noticed had the code been wrong. See :func:`mutation_gate`.
    """
    commands = plan.get("commands") or {}
    verification = Verification()
    run_commands(sandbox, [commands.get("build", "")], "build", verification, log=log)
    # Filtered here too, not just at planning: tickets created before the check
    # exists still carry prose that would fail as a command.
    runnable, _ = split_validation(ticket.validation)
    run_commands(sandbox, runnable, f"ticket:{ticket.id}", verification, log=log)
    run_commands(sandbox, [commands.get("test", "")], "tests", verification,
                 required=tests_required, log=log)
    verification.gates.append(mutation_gate(
        sandbox, plan, changed, mutation, log=log,
        tests_passed=any(g.name == "tests" and g.passed and not g.skipped
                         for g in verification.gates)))
    verification.gates.append(
        security_gate(sandbox.root, changed, classifier=classifier, clean="clean"))
    record_installs(sandbox, plan)
    return verification


def change_view(sandbox: Sandbox, change, budget: int) -> str:
    """The written files, whole while they fit, as the reviewer sees them.

    Files are kept entire in the order they were written rather than sampled
    from each: a reviewer shown the first half of every file cannot tell a
    missing error path from one it was not given.
    """
    parts, used = [], 0
    for path in change.written:
        text = sandbox.read(path)
        slice_ = text[: max(0, budget - used)]
        if not slice_:
            break
        used += len(slice_)
        parts.append(f"--- {path} ---\n{slice_}")
    if change.deleted:
        parts.append("deleted: " + ", ".join(change.deleted))
    return "\n\n".join(parts) or change.describe()


def review(agents, ticket, sandbox: Sandbox, change, verification: Verification,
           prior: str = "", toolbox: Toolbox | None = None, log: Logger = NULL) -> dict:
    """Reviewer verdict on a ticket's change. Blocking issues send it back.

    *prior* is the work journal, which lets the reviewer catch what a diff
    alone cannot: duplicating an existing component, or renaming an interface
    another ticket already depends on. It is also the first thing dropped when
    the change itself does not fit: a review of code the reviewer was not
    shown is worse than one made without the history.

    *toolbox* lets the reviewer read the rest of the workspace. A change is
    shown without the code around it, so a caller the change breaks and a
    helper it duplicates are both outside the diff; without a lookup the
    reviewer can only guess at them, and a guessed defect blocks a ticket
    that was correct.
    """
    reviewer = agents["reviewer"]
    prompt = Prompt(reviewer, log=log)
    prompt.add("ticket",
               f"TICKET {ticket.id}: {ticket.title}\n"
               f"OBJECTIVE: {ticket.objective}\n"
               "ACCEPTANCE CRITERIA:\n"
               + "\n".join(f"- {a}" for a in ticket.acceptance), required=True)
    prompt.add("journal", prior, priority=30, floor=800)
    prompt.add("gates", f"VERIFICATION RESULTS:\n{verification.report()}",
               priority=20, floor=400)
    view = prompt.add("change", priority=10, floor=2000)
    prompt.add("instructions", prompts.REVIEW, required=True)
    lookup = Lookup(reviewer, prompt, toolbox, log=log)
    header = "CHANGE:\n"
    view.fill(header + change_view(sandbox, change,
                                   max(0, view.room() - len(header))))
    verdict = lookup.ask()
    blocking = [b for b in (verdict.get("blocking") or []) if str(b).strip()]
    security = [s for s in (verdict.get("security") or []) if str(s).strip()]
    verdict["approved"] = bool(verdict.get("approved")) and not blocking and not security
    verdict["blocking"] = blocking + security
    log.info("review.verdict", ticket=ticket.id, approved=verdict["approved"],
             blocking=len(verdict["blocking"]))
    return verdict


def triage(agents, ticket, verification: Verification, context: str,
           attempts: int, log: Logger = NULL) -> dict:
    """Classify a failure and prescribe the smallest repair.

    The failure output is the one block worth keeping the end of: a build log
    puts the error last, and the diagnosis is in the error.
    """
    triager = agents["triager"]
    prompt = Prompt(triager, log=log)
    prompt.add("context", context, priority=30, floor=1000)
    prompt.add("ticket", f"TICKET {ticket.id}: {ticket.title}\n"
                         f"ATTEMPT {attempts} of this ticket.", required=True)
    prompt.add("gates", f"GATES:\n{verification.report()}", priority=10, floor=400)
    output = prompt.add("output", priority=20, floor=600, tail=True)
    prompt.add("instructions", prompts.TRIAGE, required=True)
    header = "OUTPUT:\n"
    output.fill(header + verification.failure_output(
        max(0, output.room() - len(header))))
    verdict = triager.ask_json(prompt.render())
    log.info("triage", ticket=ticket.id, classification=verdict.get("classification"),
             action=verdict.get("action"))
    return verdict


def record_evaluation(root: Path, label: str, verification: Verification,
                      extra: dict | None = None) -> Path:
    path = (Path(root) / ".hermes" / "evaluations" /
            f"{time.strftime('%Y%m%d-%H%M%S')}-{label}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "label": label,
        "at": time.time(),
        "passed": verification.passed,
        "gates": [{"name": g.name, "passed": g.passed, "skipped": g.skipped,
                   "detail": g.detail[:2000]} for g in verification.gates],
        **(extra or {}),
    }, indent=2, default=str))
    return path
