"""Mutation testing: break the code on purpose and see whether the tests notice.

The factory writes the code and then writes the tests for that same code, which
is the one configuration where a green suite proves least. Coverage does not
catch it -- a test that imports a function, calls it and asserts nothing runs
every line. The direct instrument is to inject a small fault and ask whether
the suite goes red. A fault that survives is a sentence about the tests, not
about the code: *this* change would have shipped.

The engine here is deliberately small and has no dependency. mutmut and
cosmic-ray both want to own the run -- their own config file, their own
collection of the test suite, their own cache directory inside the project --
and Hermes already owns the sandbox, the test command and the budget. What is
actually needed is an operator table and a way to write one mutant at a time
into a workspace, which is this module.

Mutants are produced through :mod:`ast` rather than by patching text, so a
mutant is always syntactically valid and always changes behaviour rather than
spelling. The cost is that the mutated file is ``ast.unparse``'d and so loses
its comments and its formatting; that file exists for one test run and is
restored afterwards, and the *report* names the statement, never the file.
"""
from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from .log import Logger, NULL
from .sandbox import Sandbox

#: Comparisons, swapped for the neighbour a test is most likely to miss: an
#: off-by-one boundary rather than a negation, where both are available.
COMPARISONS = {
    ast.Eq: (ast.NotEq, "== to !="),
    ast.NotEq: (ast.Eq, "!= to =="),
    ast.Lt: (ast.LtE, "< to <="),
    ast.LtE: (ast.Lt, "<= to <"),
    ast.Gt: (ast.GtE, "> to >="),
    ast.GtE: (ast.Gt, ">= to >"),
    ast.Is: (ast.IsNot, "is to is not"),
    ast.IsNot: (ast.Is, "is not to is"),
    ast.In: (ast.NotIn, "in to not in"),
    ast.NotIn: (ast.In, "not in to in"),
}

#: Arithmetic. ``+`` on a list or a string raises under ``-`` instead of
#: computing the wrong answer, which still kills the mutant -- a suite that
#: does not notice a TypeError is not exercising the line at all.
ARITHMETIC = {
    ast.Add: (ast.Sub, "+ to -"),
    ast.Sub: (ast.Add, "- to +"),
    ast.Mult: (ast.FloorDiv, "* to //"),
    ast.Div: (ast.Mult, "/ to *"),
    ast.FloorDiv: (ast.Mult, "// to *"),
    ast.Mod: (ast.Mult, "% to *"),
    ast.Pow: (ast.Mult, "** to *"),
    ast.LShift: (ast.RShift, "<< to >>"),
    ast.RShift: (ast.LShift, ">> to <<"),
    ast.BitAnd: (ast.BitOr, "& to |"),
    ast.BitOr: (ast.BitAnd, "| to &"),
}

#: How much of a statement a report quotes. A whole mutated line is the useful
#: unit; a mutated 400-character comprehension is not, and the location says
#: where to look for the rest.
SNIPPET = 160


@dataclass(frozen=True)
class Mutant:
    """One injected fault: where it goes, what it changes, how it reads.

    *after* is empty until the mutant has actually been applied -- rendering
    every possible mutant of a file to decide which five to run would unparse
    the file once per mutation point, and there are hundreds.
    """
    path: str
    index: int          #: which mutation point in the file, in traversal order
    operator: str       #: what changed, in the words of the operator table
    line: int
    before: str         #: the original statement
    after: str = ""     #: the same statement with the fault in it

    def describe(self) -> str:
        head = f"{self.path}:{self.line}: {self.operator}"
        if not self.after or self.after == self.before:
            return f"{head}\n  in: {self.before}" if self.before else head
        return f"{head}\n  was: {self.before}\n  now: {self.after}"


@dataclass
class Survey:
    """What a mutation run found, and what stopped it."""
    mutants: list[Mutant] = field(default_factory=list)     #: actually run
    survivors: list[Mutant] = field(default_factory=list)   #: tests stayed green
    errors: list[str] = field(default_factory=list)
    exhausted: bool = False     #: the budget ran out before the mutants did

    @property
    def killed(self) -> int:
        return len(self.mutants) - len(self.survivors)

    @property
    def score(self) -> float:
        """Killed fraction. 1.0 means every injected fault was noticed."""
        return self.killed / len(self.mutants) if self.mutants else 1.0

    def summary(self) -> str:
        if not self.mutants:
            return "no mutants generated"
        head = (f"{self.killed}/{len(self.mutants)} faults caught"
                + (" (budget exhausted)" if self.exhausted else ""))
        return "\n".join([head] + [m.describe() for m in self.survivors])


class _Mutator(ast.NodeTransformer):
    """Walks a tree offering mutations, and applies the *target*-th one.

    Counting and applying are the same traversal on purpose. Two traversals
    that numbered the opportunities differently would mean the mutant that ran
    was not the mutant that got reported, and the report is the whole product.
    """

    def __init__(self, target: int = -1) -> None:
        self.target = target
        self.points: list[tuple[str, int, str]] = []    # operator, line, statement
        self.hit: ast.stmt | None = None                # statement holding the fault
        self.statement: ast.stmt | None = None

    def _offer(self, node: ast.AST, operator: str, mutate) -> ast.AST:
        """Record a mutation point, and take it if it is the one asked for."""
        index = len(self.points)
        line = getattr(node, "lineno", 0)
        self.points.append((operator, line, _quote(self.statement)))
        if index != self.target or self.hit is not None:
            return node
        self.hit = self.statement
        return mutate() or node

    def visit(self, node):
        if isinstance(node, ast.stmt):
            outer, self.statement = self.statement, node
            try:
                return super().visit(node)
            finally:
                self.statement = outer
        return super().visit(node)

    # -- the operators ---------------------------------------------------
    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        for position, op in enumerate(node.ops):
            swap = COMPARISONS.get(type(op))
            if swap:
                self._offer(node, swap[1], _setter(node.ops, position, swap[0]()))
        return node

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        swap = ARITHMETIC.get(type(node.op))
        return self._offer(node, swap[1], _attr(node, "op", swap[0]())) if swap else node

    def visit_AugAssign(self, node: ast.AugAssign):
        self.generic_visit(node)
        swap = ARITHMETIC.get(type(node.op))
        return self._offer(node, swap[1].replace(" to ", "= to ") + "=",
                           _attr(node, "op", swap[0]())) if swap else node

    def visit_BoolOp(self, node: ast.BoolOp):
        self.generic_visit(node)
        swap = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        name = "and to or" if isinstance(node.op, ast.And) else "or to and"
        return self._offer(node, name, _attr(node, "op", swap))

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return self._offer(node, "dropped a not", lambda: node.operand)
        return node

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool):
            return self._offer(node, f"{node.value} to {not node.value}",
                               _attr(node, "value", not node.value))
        # Strings are left alone: mutating one is as likely to change a log
        # line as a decision, and a suite that asserts on log text is not what
        # this is measuring.
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return self._offer(node, f"{node.value!r} to {node.value + 1!r}",
                               _attr(node, "value", node.value + 1))
        return node

    def visit_Return(self, node: ast.Return):
        self.generic_visit(node)
        if node.value is None or _is_none(node.value):
            return node
        return self._offer(node, "returned None instead", _attr(node, "value", None))


def _setter(items: list, position: int, value):
    return lambda: (items.__setitem__(position, value), None)[1]


def _attr(node: ast.AST, name: str, value):
    return lambda: (setattr(node, name, value), None)[1]


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _quote(statement: ast.stmt | None) -> str:
    """One statement as a reader would say it, without its body.

    A mutation inside an ``if`` test should quote the test, not the fifty
    lines the branch guards, so compound statements are cut at their header.
    """
    if statement is None:
        return ""
    try:
        text = ast.unparse(statement)
    except Exception:           # unparse is best-effort on an exotic tree
        return ""
    return text.split("\n", 1)[0].strip()[:SNIPPET]


def points(source: str) -> list[tuple[str, int, str]]:
    """Every mutation this module can make in *source*, in traversal order."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    mutator = _Mutator()
    mutator.visit(tree)
    return mutator.points


def mutants(path: str, source: str) -> list[Mutant]:
    """Describe every available mutant of *source* without applying any."""
    return [Mutant(path, index, operator, line, before)
            for index, (operator, line, before) in enumerate(points(source))]


def apply(source: str, index: int) -> tuple[str, str] | None:
    """Return *source* with mutation *index* applied, plus the changed statement.

    ``None`` when the mutation cannot be made or cannot be rendered -- an
    unparseable file, an index past the end, or a tree ``ast.unparse`` refuses.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    mutator = _Mutator(index)
    tree = mutator.visit(tree)
    if mutator.hit is None:
        return None
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree), _quote(mutator.hit)
    except Exception:
        return None


def select(candidates: list[Mutant], limit: int) -> list[Mutant]:
    """Spread *limit* picks over the files and over each file's mutation points.

    Taking the first N would test the top of one file N times, which is where
    the imports and the constants are. The spread is arithmetic rather than
    random so that two runs over the same change inject the same faults: a
    mutation report nobody can reproduce is a rumour.
    """
    if limit <= 0 or not candidates:
        return []
    by_path: dict[str, list[Mutant]] = {}
    for mutant in candidates:
        by_path.setdefault(mutant.path, []).append(mutant)
    share = max(1, limit // len(by_path))
    picked: list[Mutant] = []
    for group in by_path.values():
        step = max(1, len(group) // share)
        picked.extend(group[::step][:share])
    picked.sort(key=lambda m: (m.path, m.index))
    return picked[:limit]


def candidates(sandbox: Sandbox, paths: list[str]) -> list[Mutant]:
    """Mutants available in the Python sources among *paths*.

    Test files are excluded: a fault injected into a test proves nothing about
    whether the tests would catch a fault.
    """
    found: list[Mutant] = []
    for path in sorted(set(paths)):
        name = Path(path).name
        if not path.endswith(".py") or name.startswith("test_") or name.endswith("_test.py"):
            continue
        if "tests" in Path(path).parts[:-1]:
            continue
        source = sandbox.read(path)
        if source:
            found.extend(mutants(path, source))
    return found


def survey(sandbox: Sandbox, paths: list[str], command: str, *, limit: int = 5,
           budget: int = 180, timeout: int = 120, log: Logger = NULL) -> Survey:
    """Inject faults one at a time and record which ones the suite ignored.

    The workspace is left exactly as it was found. Every mutant is written,
    tested and reverted before the next is written, so a crash mid-survey
    costs one restore, not a corrupted tree -- and two mutants never interact,
    which would make a survivor unattributable.
    """
    result = Survey()
    chosen = select(candidates(sandbox, paths), limit)
    if not chosen:
        return result
    deadline = time.time() + budget
    for mutant in chosen:
        if time.time() >= deadline:
            result.exhausted = True
            break
        original = sandbox.read(mutant.path)
        mutated = apply(original, mutant.index)
        if mutated is None:
            result.errors.append(f"{mutant.path}: mutation {mutant.index} did not apply")
            continue
        try:
            sandbox.write(mutant.path, mutated[0])
            run = sandbox.run(command, timeout=min(timeout, max(1, int(deadline - time.time()))))
        finally:
            sandbox.write(mutant.path, original)
        mutant = replace(mutant, after=mutated[1])
        result.mutants.append(mutant)
        if run.ok:
            result.survivors.append(mutant)
            log.info("mutation.survived", path=mutant.path, line=mutant.line,
                     operator=mutant.operator)
        else:
            log.debug("mutation.killed", path=mutant.path, line=mutant.line,
                      operator=mutant.operator)
    log.info("mutation.survey", tested=len(result.mutants),
             survived=len(result.survivors), exhausted=result.exhausted)
    return result
