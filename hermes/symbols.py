"""What the project defines, and where each name is used, from its own parser.

An agent that wants to call something it did not write has three ways to find
out what it is called. It can be shown the file, which costs the file. It can
grep, which costs a pattern guess and returns every line that happens to
contain the word -- the import, the comment, the docstring, the call -- with
no way to tell the definition from the mentions. Or it can ask a question
that has a name in it: *where is this defined*, *who uses it*, *what does
this file define*. Only the third is cheap in tokens and exact in the answer,
and it is the one nothing in the toolbox could ask before this module.

The index is built with :mod:`ast`, the interpreter's own parser, and so it
covers Python and says so. That is a deliberate limit rather than an
oversight: a regular expression over C++ or Rust would answer the same
questions at a confidence the answer does not carry, and an agent that is
told "``Widget`` is defined at engine.cpp:120" has no way to discover the
line was a forward declaration inside a comment. A project in another
language keeps :func:`~hermes.tools.Toolbox._grep`, which never claimed to
understand anything. The one thing an index must not do is be wrong
authoritatively.

Two kinds of row come out, and the asymmetry between them is the point:

* a **definition** is exact -- the parser says this is a class, this is its
  line span, these are its arguments, this is the first line of its
  docstring;
* a **reference** is a name used at a line, and nothing resolves it. Two
  classes with a ``run`` method produce references indistinguishable here.

So :func:`parse` records what the syntax tree knows and the tools that serve
it say which of the two an answer came from. A reference list is a place to
look, a definition is a fact.

A file that does not parse is recorded as not having parsed, with the line
that broke, and its old symbols are dropped rather than kept. Stale
definitions are worse than none: an agent shown a signature that no longer
exists spends a generation writing against it.
"""
from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
from typing import Iterator

#: What this module reads. Everything else is left to grep.
SUFFIXES = (".py", ".pyi")

#: Names that are in every module without anyone importing them, and so say
#: nothing about who depends on what. ``self`` and ``cls`` are here for the
#: same reason: every method mentions them.
IGNORED = frozenset(dir(builtins)) | {"self", "cls", "_"}

#: What a definition can be. Ordered by how much a reader usually wants it
#: first, which is the order an answer lists them in.
KINDS = ("class", "function", "method", "constant", "field")

#: Longest signature and docstring line kept. Both are there to be read in a
#: list of results, not to replace reading the file.
SIGNATURE_CHARS = 200
DOC_CHARS = 160


@dataclass
class Definition:
    """One thing a file defines, as its own parser saw it."""

    path: str = ""
    name: str = ""
    #: Dotted path within the file: ``Toolbox.round``. What makes a method
    #: findable by the class that owns it.
    qualname: str = ""
    kind: str = "function"
    line: int = 0
    end_line: int = 0
    signature: str = ""
    doc: str = ""
    #: The enclosing qualname, or empty at module level.
    parent: str = ""

    @property
    def lines(self) -> int:
        return max(1, self.end_line - self.line + 1)

    def render(self) -> str:
        head = f"{self.path}:{self.line}  {self.signature or self.qualname}"
        return f"{head}\n    {self.doc}" if self.doc else head


@dataclass
class Reference:
    """A name used at a line. Not resolved, and never presented as resolved."""

    path: str = ""
    name: str = ""
    #: The definition the use sits inside, or empty at module level.
    qualname: str = ""
    line: int = 0
    #: call, import, base, decorator, attribute, name.
    kind: str = "name"
    #: Whatever else the syntax knew: the module an import came from.
    detail: str = ""

    def render(self) -> str:
        where = f" in {self.qualname}" if self.qualname else ""
        detail = f" (from {self.detail})" if self.detail else ""
        return f"{self.path}:{self.line}  {self.kind}{where}{detail}"


@dataclass
class Parse:
    """What came out of one file, including the case where nothing did."""

    path: str = ""
    defs: list[Definition] = field(default_factory=list)
    refs: list[Reference] = field(default_factory=list)
    #: True only when this is a file the index reads and it parsed.
    parsed: bool = False
    #: Why there is nothing, when there is nothing. An agent asking for an
    #: outline of a broken file is better served by the syntax error than by
    #: an empty list that looks like an empty file.
    error: str = ""


def indexes(path: str) -> bool:
    """Whether this module claims to understand the file at *path*."""
    return str(path).endswith(SUFFIXES)


def parse(path: str, text: str) -> Parse:
    """Definitions and references in *text*, or the reason there are none."""
    if not indexes(path):
        return Parse(path=path, error="not a Python file")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        where = f" at line {exc.lineno}" if exc.lineno else ""
        return Parse(path=path, error=f"{exc.msg}{where}")
    except (ValueError, RecursionError) as exc:
        # Null bytes, or a nesting depth CPython's own parser gives up on.
        return Parse(path=path, error=str(exc)[:120] or type(exc).__name__)
    walker = _Walker(str(path))
    walker.visit(tree)
    return Parse(path=str(path), defs=walker.defs, refs=walker.refs, parsed=True)


class _Walker(ast.NodeVisitor):
    """Walks a module, carrying the scope each node sits in.

    Scope is what turns a name into an answer: a reference is only useful if
    it says which function makes it, and a method is only findable if it
    remembers its class. Everything here is lexical -- the enclosing ``def``
    and ``class`` statements, nothing about what any name binds to.
    """

    def __init__(self, path: str):
        self.path = path
        self.defs: list[Definition] = []
        self.refs: list[Reference] = []
        self._scope: list[tuple[str, str]] = []

    # -- scope -----------------------------------------------------------
    @property
    def scope(self) -> str:
        return ".".join(name for name, _ in self._scope)

    @property
    def in_class(self) -> bool:
        return bool(self._scope) and self._scope[-1][1] == "class"

    def _qualify(self, name: str) -> str:
        return f"{self.scope}.{name}" if self.scope else name

    # -- definitions -----------------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = self._qualify(node.name)
        self._define(node, qualname, "class", _class_signature(node))
        for base in node.bases:
            self._uses(base, self.scope, forced="base")
        for decorator in node.decorator_list:
            self._uses(decorator, self.scope, forced="decorator")
        self._scope.append((node.name, "class"))
        for statement in node.body:
            self.visit(statement)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node) -> None:
        qualname = self._qualify(node.name)
        kind = "method" if self.in_class else "function"
        self._define(node, qualname, kind, _function_signature(node))
        # Decorators, defaults and annotations are evaluated where the `def`
        # stands, not inside it, so they are recorded against the outer scope.
        for decorator in node.decorator_list:
            self._uses(decorator, self.scope, forced="decorator")
        for value in _annotations(node):
            self._uses(value, self.scope)
        self._scope.append((node.name, "function"))
        for statement in node.body:
            self.visit(statement)
        self._scope.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._bind(node, target.id, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self._bind(node, node.target.id, node.value, node.annotation)
        self.generic_visit(node)

    def _bind(self, node, name: str, value, annotation=None) -> None:
        """Record a module constant or a class field; ignore locals.

        Inside a function every assignment is a local, and a local is not
        something another file can refer to. At class level it is a field --
        which is how this factory's dataclasses say what they hold, and so
        the question an outline most often has to answer.
        """
        if self.in_class:
            kind = "field"
        elif not self._scope:
            kind = "constant"
        else:
            return
        signature = f"{name}: {_source(annotation)}" if annotation else name
        if value is not None:
            signature = f"{signature} = {_source(value)}"
        self.defs.append(Definition(
            path=self.path, name=name, qualname=self._qualify(name), kind=kind,
            line=node.lineno, end_line=_line(node),
            signature=signature[:SIGNATURE_CHARS], parent=self.scope))

    def _define(self, node, qualname: str, kind: str, signature: str) -> None:
        self.defs.append(Definition(
            path=self.path, name=node.name, qualname=qualname, kind=kind,
            line=node.lineno, end_line=_line(node),
            signature=signature[:SIGNATURE_CHARS],
            doc=_docline(node), parent=self.scope))

    # -- references ------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.refs.append(Reference(
                path=self.path, name=alias.name.rsplit(".", 1)[-1],
                qualname=self.scope, line=node.lineno, kind="import",
                detail=alias.name))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * (node.level or 0) + (node.module or "")
        for alias in node.names:
            self.refs.append(Reference(
                path=self.path, name=alias.name, qualname=self.scope,
                line=node.lineno, kind="import", detail=module))

    # The four below are where every expression ends up. They hand the whole
    # subtree to :meth:`_uses` and do not walk on, because `_uses` walks it
    # itself -- which is what lets it see a call and its callee together and
    # so report the callee once, as a call.
    def visit_Expr(self, node: ast.Expr) -> None:
        self._uses(node.value, self.scope)

    def visit_Call(self, node: ast.Call) -> None:
        self._uses(node, self.scope)

    def visit_Name(self, node: ast.Name) -> None:
        self._uses(node, self.scope)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._uses(node, self.scope)

    def _uses(self, node: ast.AST, scope: str, forced: str = "") -> None:
        for name, kind, line in _names(node):
            if name in IGNORED:
                continue
            self.refs.append(Reference(path=self.path, name=name, qualname=scope,
                                       line=line, kind=forced or kind))


def _names(node: ast.AST) -> Iterator[tuple[str, str, int]]:
    """Every name an expression uses, and how it used it.

    The subtlety is the callee: ``run(x)`` holds a ``Name`` inside a ``Call``,
    and reporting it twice would make every call look like a call and a
    mention. ``ast.walk`` yields a parent before its children, so the callee
    is known to be one by the time it comes round.

    The line is the end of the name rather than the start of the expression
    holding it. They differ whenever a call is written across lines, and the
    one worth reporting is the one the reader will find the word on.
    """
    called: set[int] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            called.add(id(sub.func))
            name = _called(sub.func)
            if name:
                yield name, "call", _line(sub.func)
        elif isinstance(sub, ast.Name):
            if id(sub) not in called and isinstance(sub.ctx, ast.Load):
                yield sub.id, "name", sub.lineno
        elif isinstance(sub, ast.Attribute):
            if id(sub) not in called and isinstance(sub.ctx, ast.Load):
                yield sub.attr, "attribute", _line(sub)


def _line(node: ast.AST) -> int:
    """The line a name is actually written on."""
    return getattr(node, "end_lineno", None) or node.lineno


def _called(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _annotations(node) -> Iterator[ast.AST]:
    """Argument annotations, defaults and the return type of a function.

    A type is a dependency: ``def make(w: Widget) -> Result`` is two more
    answers to "who uses Widget" than the body alone would give.
    """
    args = node.args
    for group in (args.posonlyargs, args.args, args.kwonlyargs):
        for arg in group:
            if arg.annotation is not None:
                yield arg.annotation
    for extra in (args.vararg, args.kwarg):
        if extra is not None and extra.annotation is not None:
            yield extra.annotation
    for default in list(args.defaults) + [d for d in args.kw_defaults if d]:
        yield default
    if node.returns is not None:
        yield node.returns


def _function_signature(node) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {_source(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({_source(node.args)}){returns}"


def _class_signature(node: ast.ClassDef) -> str:
    bases = [_source(base) for base in node.bases]
    bases += [f"{kw.arg}={_source(kw.value)}" for kw in node.keywords if kw.arg]
    return f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"


def _source(node: ast.AST | None) -> str:
    """One node as the code that would produce it, or a placeholder.

    ``ast.unparse`` is exact for everything this asks it about, but it is not
    total; a signature is a convenience and must never be the thing that
    fails an index.
    """
    if node is None:
        return ""
    try:
        return " ".join(ast.unparse(node).split())
    except Exception:  # noqa: BLE001 - an unprintable annotation is not an error
        return "…"


def _docline(node) -> str:
    doc = ast.get_docstring(node) or ""
    first = doc.strip().split("\n", 1)[0].strip()
    return first[:DOC_CHARS]
