"""Test tampering: a change that turns the suite green by asking it less.

The implementer may change the tests in the same change as the code, and the
gates then run the tests it changed. Telling a model not to is measured to
barely help -- BaitBench saw 57% of runs by seven frontier agents take a
planted shortcut, and an instruction against it cut that by six points -- and
the strategies it takes are few and mechanical: delete the failing assertion,
rewrite it to match what the code does, skip the test, or reach into the test
settings so the test is never collected at all. Each of those leaves a trace
in the tests themselves, which is what this module reads.

The comparison is against the tests as they stood at the last *completed*
ticket, not at the start of this attempt. A failed attempt leaves its files
in place for the next one to build on, so a baseline taken per attempt would
let the first attempt weaken the suite and the second pass against the
weakened copy. That baseline is kept in :data:`BASELINE` so a resumed run
compares against the same tree.

Two kinds of evidence, of different strength:

* Structural, from :mod:`ast`: every check the tests make (``assert``, a
  ``pytest.raises``, a ``self.assert*``, a ``raise AssertionError``), every
  test function, every skip or xfail marker, the shared setup in
  ``conftest.py`` and the pytest section of each settings file. A check or a
  test that moved -- into a helper, into
  another file, under a new name -- is found where it went and is not
  reported. What is reported is a *soft* finding: a fact about the edit, not
  yet a verdict on it, because a ticket that changes behaviour legitimately
  changes the check for it.
* The collected test ids, from the project's own test command with
  ``--collect-only``. That is the one view that sees what the settings do: an
  ``addopts`` that deselects, a hook that drops items, a ``collect_ignore``.
  Fewer tests than before is a *hard* finding for any ticket that is not a
  :data:`MAY_SHRINK` kind, since no implementation ticket needs the suite to
  get smaller: an obsolete test is replaced, not deleted.

Python only, like :mod:`hermes.mutate`. Other stacks have no finding to make
here, and the gate says so rather than passing silently.
"""
from __future__ import annotations

import ast
import difflib
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

from .contextdb import SKIP
from .log import Logger, NULL
from .sandbox import Sandbox

#: Where the baseline lives between tickets, so a resumed run is compared
#: against the tree its last completed ticket left and not whatever a failed
#: ticket was half way through.
BASELINE = Path(".hermes") / "state" / "tests.json"

#: Ticket kinds whose job can make the suite smaller: the tester rewriting the
#: tests it owns, a refactor merging two tests into one. For these a lost test
#: is a question for the screen rather than a refusal.
MAY_SHRINK = ("test", "refactor")

#: Settings files pytest reads, and the section of each that is pytest's.
#: Only that section is compared: a new dependency in ``pyproject.toml`` is
#: not a change to what the tests check.
SETTINGS = {
    "pytest.ini": ("pytest",), ".pytest.ini": ("pytest",), "tox.ini": ("pytest",),
    "setup.cfg": ("tool:pytest",), "pytest.toml": ("pytest",),
    ".pytest.toml": ("pytest",),
    "pyproject.toml": ("tool.pytest.ini_options", "tool.pytest"),
}

#: A marker or call that stops a test from running or from failing.
SKIPS = re.compile(r"\b(?:skip|skipif|skipIf|skipUnless|xfail|expectedFailure)\b")
SKIP_CALLS = frozenset({"skip", "xfail", "skipTest", "importorskip"})

#: Calls that check something, besides ``assert``. A name starting ``assert``
#: covers unittest's methods and the ``assert_*`` helpers numpy and pandas ship.
CHECK_CALLS = frozenset({"raises", "warns", "deprecated_call", "fail"})

#: What in a ``conftest.py`` can change which tests run, as opposed to what a
#: fixture hands them: a hook, an ignore list, a fixture every test gets.
HOOKS = re.compile(r"^pytest_\w+$|^collect_ignore(?:_glob)?$")

#: How much of a check a finding quotes.
QUOTE = 120


def is_test(path: str) -> bool:
    """Whether *path* is part of the Python test suite rather than the code.

    One answer for both sides of the fence: :mod:`hermes.mutate` never
    injects a fault here, because a broken test proves nothing about whether
    tests catch breakage, and this module compares exactly these files.
    """
    parts = PurePosixPath(str(path).replace("\\", "/"))
    if parts.suffix != ".py":
        return False
    return (parts.name.startswith("test_") or parts.name.endswith("_test.py")
            or parts.name == "conftest.py"
            or any(part in ("tests", "test") for part in parts.parts[:-1]))


def collect_command(test_command: str) -> str:
    """The test command, asked to list its tests instead of running them.

    Empty unless the command ends in one plain pytest run: the flags are
    appended, so they have to land on pytest and not on whatever runs after
    it, and output that is piped or redirected is not output this can read.
    ``--verbosity=-1`` is set rather than another ``-q``, because verbosity is
    a count and a second ``-q`` on top of the project's own prints per-file
    totals instead of the ids.
    """
    command = (test_command or "").strip()
    if not command or re.search(r"[|<>`]|\$\(", command):
        return ""
    last = re.split(r"&&|;", command)[-1].strip()
    if not re.match(r"(?:\w+=\S*\s+)*(?:\S*python3?(?:\.\d+)?\s+-m\s+pytest|pytest|py\.test)\b",
                    last):
        return ""
    return f"{command} --collect-only --verbosity=-1"


@dataclass
class Surface:
    """The tests as the gates see them, at one moment.

    *collected* is None when the ids could not be read, and *note* says why:
    no pytest command, or one that did not collect. An empty list is a suite
    that collected nothing, which is a different fact.
    """

    files: dict[str, str] = field(default_factory=dict)      #: test sources
    settings: dict[str, str] = field(default_factory=dict)   #: pytest sections
    collected: list[str] | None = None
    command: str = ""
    note: str = ""

    @property
    def empty(self) -> bool:
        return not self.files and not self.collected

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Surface":
        data = json.loads(text)
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass(frozen=True)
class Finding:
    """One way the tests now ask less, and where."""

    where: str
    what: str
    hard: bool = False

    def describe(self) -> str:
        return f"{self.where} {self.what}"


def capture(sandbox: Sandbox, command: str, *, timeout: int = 120,
            log: Logger = NULL) -> Surface:
    """Read the test surface off the workspace, collecting with *command*.

    *command* is a :func:`collect_command`, fixed when the baseline was taken
    and reused for every comparison against it. Collection with two different
    commands would count two different suites.
    """
    surface = Surface(command=command)
    for rel, path in _walk(Path(sandbox.root)):
        name = PurePosixPath(rel).name
        try:
            if is_test(rel):
                surface.files[rel] = path.read_text(errors="replace")
            elif name in SETTINGS:
                section = _section(path.read_text(errors="replace"), SETTINGS[name])
                if section:
                    surface.settings[rel] = section
        except OSError:
            continue
    if not command:
        surface.note = "the test command is not a single pytest run"
        return surface
    result = sandbox.run(command, timeout=timeout)
    if result.exit_code in (0, 5):          # 5: pytest collected nothing
        surface.collected = _ids(result.stdout)
    else:
        surface.note = f"collection failed: {result.summary(400)}"
    log.debug("tamper.captured", files=len(surface.files),
              collected=-1 if surface.collected is None else len(surface.collected))
    return surface


def compare(before: Surface, after: Surface, kind: str = "implementation") -> list[Finding]:
    """Every way *after* asks less of the code than *before* did.

    Additions are never findings. Nothing is weakened by a new test, and a
    new file that deselects old ones shows up where it matters, in the ids.
    """
    if before.empty:
        return []
    findings: list[Finding] = []
    old, new = _parse(before.files), _parse(after.files)
    for path in sorted(set(old) & (set(after.files) - set(new))):
        findings.append(Finding(path, "no longer parses, so none of its tests can run"))
        del old[path]
    moved = {shape for module in new.values() for shape in module.shapes}
    gone: set[tuple[str, str]] = set()
    for path, module in old.items():
        current = new.get(path)
        for name, test in module.tests.items():
            if current is not None and name in current.tests:
                added = current.tests[name].markers - test.markers
                if added:
                    findings.append(Finding(f"{path}::{name}", "is now skipped or "
                                            f"expected to fail: `{_quote(min(added))}`"))
            elif test.shape not in moved:
                gone.add((path, name))
                findings.append(Finding(f"{path}::{name}", "was deleted"))
        if current is not None and current.markers - module.markers:
            findings.append(Finding(path, "now skips every test in it: "
                                    f"`{_quote(min(current.markers - module.markers))}`"))
    findings += _lost_checks(old, new, gone)
    findings += _setup_changes(old, new)
    for path in sorted(set(before.settings) | set(after.settings)):
        if _norm(before.settings.get(path, "")) != _norm(after.settings.get(path, "")):
            header = (after.settings.get(path) or before.settings[path]).splitlines()[0]
            findings.append(Finding(path, f"has different pytest settings under `{header}`"))
    findings += _collection(before, after, kind)
    return findings


def evidence(before: Surface, after: Surface, findings: list[Finding],
             budget: int = 6000) -> str:
    """The part of the change the findings are about, as unified diffs.

    Only the files a finding names: the screen is asked about the tests, and
    a diff of the code beside them is what would let it talk itself round.
    """
    paths = list(dict.fromkeys(f.where.split("::", 1)[0] for f in findings))
    chunks, used = [], 0
    for path in paths:
        if path in before.settings or path in after.settings:
            old, new = before.settings.get(path, ""), after.settings.get(path, "")
        elif path in before.files:
            old, new = before.files[path], after.files.get(path, "")
        else:
            continue
        diff = "".join(difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            f"a/{path}", f"b/{path}" if new else "/dev/null"))
        piece = diff[: max(0, budget - used)]
        if not piece:
            break
        used += len(piece)
        chunks.append(piece)
    return "\n".join(chunks)


def load(root: Path) -> Surface | None:
    path = Path(root) / BASELINE
    try:
        return Surface.from_json(path.read_text())
    except (OSError, ValueError, TypeError):
        return None


def save(root: Path, surface: Surface) -> None:
    path = Path(root) / BASELINE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(surface.to_json())


def forget(root: Path) -> None:
    """A ticket completed: its tests are the ones the next ticket answers to."""
    (Path(root) / BASELINE).unlink(missing_ok=True)


# -- reading the tests ---------------------------------------------------------

@dataclass(frozen=True)
class _Test:
    shape: str                  #: the body alone, so a renamed or moved test is found
    markers: frozenset[str]     #: skip and xfail markers, the class's included


@dataclass
class _Module:
    tests: dict[str, _Test] = field(default_factory=dict)
    checks: list[tuple[str, int]] = field(default_factory=list)
    spans: list[tuple[int, int, str]] = field(default_factory=list)
    shapes: set[str] = field(default_factory=set)
    markers: frozenset[str] = frozenset()
    #: Top-level statements as (dump, label, whether it steers collection).
    statements: list[tuple[str, str, bool]] = field(default_factory=list)


def _walk(root: Path):
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if d not in SKIP)
        for name in sorted(names):
            path = Path(directory) / name
            yield path.relative_to(root).as_posix(), path


def _parse(files: dict[str, str]) -> dict[str, _Module]:
    parsed = {}
    for path, text in files.items():
        try:
            parsed[path] = _read(ast.parse(text))
        except (SyntaxError, ValueError):
            continue
    return parsed


def _read(tree: ast.Module) -> _Module:
    module = _Module()
    module.checks = [(ast.unparse(node), node.lineno) for node in ast.walk(tree)
                     if isinstance(node, ast.Assert)
                     or (isinstance(node, ast.Call) and _checks(node.func))
                     or (isinstance(node, ast.Raise) and _fails(node))]
    marks: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module.statements.append((ast.dump(node), node.name, _steers(node)))
        else:
            label = ast.unparse(node).splitlines()[0][:80]
            module.statements.append((ast.dump(node), label, _steers(node)))
            marks |= _pytestmark(node) | _skip_calls(node)
    module.markers = frozenset(marks)
    _definitions(tree.body, [], frozenset(), module)
    return module


def _definitions(body, prefix: list[str], inherited: frozenset[str],
                 module: _Module) -> None:
    for node in body:
        if isinstance(node, ast.ClassDef):
            marks = set(inherited) | _decorators(node)
            for statement in node.body:
                marks |= _pytestmark(statement)
            _definitions(node.body, prefix + [node.name], frozenset(marks), module)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = "::".join(prefix + [node.name])
            shape = ast.dump(ast.Module(body=node.body, type_ignores=[]))
            module.spans.append((node.lineno, node.end_lineno or node.lineno, name))
            module.shapes.add(shape)
            if node.name.startswith("test"):
                marks = inherited | _decorators(node) | _skip_calls(node)
                module.tests[name] = _Test(shape, frozenset(marks))


def _callee(func: ast.AST) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _checks(func: ast.AST) -> bool:
    name = _callee(func)
    return name.startswith("assert") or name in CHECK_CALLS


def _fails(node: ast.Raise) -> bool:
    """``raise AssertionError(...)``: the check a test script writes by hand."""
    exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
    return exc is not None and _callee(exc) == "AssertionError"


def _decorators(node) -> set[str]:
    return {text for text in map(ast.unparse, node.decorator_list) if SKIPS.search(text)}


def _pytestmark(node: ast.AST) -> set[str]:
    if (isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)):
        text = ast.unparse(node.value)
        return {text} if SKIPS.search(text) else set()
    return set()


def _skip_calls(node: ast.AST) -> set[str]:
    return {ast.unparse(n) for n in ast.walk(node)
            if isinstance(n, ast.Call) and _callee(n.func) in SKIP_CALLS}


def _steers(node: ast.AST) -> bool:
    """Whether a conftest statement decides which tests run, not how they run."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return bool(HOOKS.match(node.name)) or any(
            "autouse" in ast.unparse(d) for d in node.decorator_list)
    targets = (node.targets if isinstance(node, ast.Assign)
               else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign))
               else [])
    return any(isinstance(t, ast.Name) and HOOKS.match(t.id) for t in targets)


# -- comparing them ------------------------------------------------------------

def _lost_checks(old: dict[str, _Module], new: dict[str, _Module],
                 gone: set[tuple[str, str]]) -> list[Finding]:
    """Checks that exist nowhere in the tests any more.

    Counted over the whole suite rather than per test, so a check moved into
    a helper or another file is still a check. A deleted test is reported
    once, as deleted, and not again for every assert it held.
    """
    lost = (Counter(text for module in old.values() for text, _ in module.checks)
            - Counter(text for module in new.values() for text, _ in module.checks))
    kept = []
    for path, module in old.items():
        for text, line in module.checks:
            owner = _owner(module, line)
            if (path, owner) in gone:
                # Charged to the deleted test first, so that when a surviving
                # test holds the same assert it is not blamed for losing it.
                lost[text] -= 1
            else:
                kept.append((path, owner, text))
    findings = []
    for path, owner, text in kept:
        if lost[text] > 0:
            lost[text] -= 1
            where = f"{path}::{owner}" if owner else path
            findings.append(Finding(where, f"no longer checks `{_quote(text)}`"))
    return findings


def _setup_changes(old: dict[str, _Module], new: dict[str, _Module]) -> list[Finding]:
    """A shared setup that lost a definition, or gained one that steers collection."""
    findings = []
    for path in sorted(set(old) | set(new)):
        if PurePosixPath(path).name != "conftest.py" or path not in new:
            continue
        before = old[path].statements if path in old else []
        dumps_before = {dump for dump, _, _ in before}
        dumps_after = {dump for dump, _, _ in new[path].statements}
        changed = [label for dump, label, _ in before if dump not in dumps_after]
        steering = [label for dump, label, steers in new[path].statements
                    if steers and dump not in dumps_before]
        parts = ([f"removed or rewrote `{_quote(', '.join(changed))}`"] if changed else []) + (
            [f"added `{_quote(', '.join(steering))}`"] if steering else [])
        if parts:
            findings.append(Finding(path, "; ".join(parts)))
    for path in sorted(set(old) - set(new)):
        if PurePosixPath(path).name == "conftest.py":
            findings.append(Finding(path, "was deleted"))
    return findings


def _collection(before: Surface, after: Surface, kind: str) -> list[Finding]:
    if before.collected is None:
        return []
    if after.collected is None:
        return [Finding("the suite", "was collected at the last completed ticket and "
                        f"cannot be now: {after.note}", hard=True)]
    lost = len(before.collected) - len(after.collected)
    if lost <= 0:
        return []
    missing = sorted(set(before.collected) - set(after.collected))
    shown = ", ".join(missing[:8]) + (f" and {len(missing) - 8} more" if len(missing) > 8 else "")
    noun = "test is" if lost == 1 else "tests are"
    return [Finding("the suite", f"is smaller: {lost} fewer {noun} collected "
                    f"({len(after.collected)}, was {len(before.collected)})"
                    + (f"; gone: {shown}" if shown else ""),
                    hard=kind not in MAY_SHRINK)]


def _owner(module: _Module, line: int) -> str:
    """The innermost definition holding *line*, or "" at module level."""
    holding = [(end - start, name) for start, end, name in module.spans
               if start <= line <= end]
    return min(holding)[1] if holding else ""


def _ids(output: str) -> list[str]:
    """Node ids from ``--collect-only`` output.

    The ids come first and end at the first blank line. Anything after it --
    a warnings summary groups its lines under node ids too -- is not a test.
    """
    ids = []
    for line in output.splitlines():
        if not line.strip():
            if ids:
                break
            continue
        if "::" in line and not line[:1].isspace():
            ids.append(line.strip())
    return ids


def _section(text: str, names: tuple[str, ...]) -> str:
    """The ``[name]`` section of an ini or toml file, header line included."""
    lines, keep = [], False
    for line in text.splitlines():
        header = re.match(r"^\[([^\[\]]+)\]\s*(?:#.*)?$", line)
        if header:
            keep = header.group(1).strip().strip('"') in names
            if keep:
                lines.append(line.strip())
            continue
        if line.startswith("[["):
            keep = False
        elif keep:
            lines.append(line)
    return "\n".join(lines)


def _norm(section: str) -> list[str]:
    return [line.strip() for line in section.splitlines()
            if line.strip() and not line.strip().startswith(("#", ";"))]


def _quote(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= QUOTE else text[: QUOTE - 1] + "…"
