"""Which functions a property can be written about, from the project's own parser.

The factory writes the code and then writes that code's tests, so an example
test agrees with the implementation because one model produced both, not
because either is right. :mod:`hermes.mutate` attacks that from one side by
breaking the code and asking whether the suite notices. A property attacks it
from the other: ``decode(encode(x)) == x`` is a claim about every ``x``, and
the inputs come from a strategy rather than from whoever wrote the function.

Only some functions can carry one. A property quantifies over inputs and
compares results, so it needs a function whose answer depends on its arguments
and on nothing else -- called twice it answers twice the same, and calling it
leaves nothing behind. Finding those is what this module does, with :mod:`ast`
and no dependency, over the sources a test ticket is about.

The rules below reject rather than admit, because the two mistakes do not cost
the same. A pure function passed over costs a property nobody wrote. An impure
one offered up costs a property wrapped around a clock or a socket, which fails
one run in twenty on a machine nobody is watching -- and the factory reads a
newly red gate as a regression of the ticket it is on, so it would then repair
code that was never broken.

What this cannot see is a call out of the file: ``save(record)`` imported from
a sibling module is a name, and nothing here resolves it. Within one file the
check is transitive -- a function calling a rejected function is rejected --
but across files it is a filter and not a proof, which is why the prompt built
from it (:func:`hermes.prompts.property_tests`) tells the Tester to drop any
function that turns out to touch the world.

Python only, for the reason :mod:`hermes.symbols` gives at greater length: a
regular expression over C++ would answer this question at a confidence the
answer does not carry. A project in another language is simply offered no
properties, and keeps the example tests it would have had anyway.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable, Sequence

#: What this module reads. Matches :data:`hermes.symbols.SUFFIXES`.
SUFFIXES = (".py", ".pyi")

#: Builtins whose call reaches outside the call, or whose answer differs
#: between runs of the same program. ``id`` and ``hash`` are the second kind
#: and are the worse one: a property over a string's hash passes until the
#: interpreter is started again with a different seed.
IMPURE_BUILTINS = frozenset({
    "open", "print", "input", "exec", "eval", "compile", "__import__",
    "breakpoint", "globals", "locals", "vars", "id", "hash", "exit", "quit",
    "help", "setattr", "delattr",
})

#: Modules that reach the filesystem, the network, the clock or the process.
#: A call on any of them disqualifies the function making it, whether it was
#: reached as ``random.shuffle`` or imported as ``shuffle`` -- the import table
#: in :func:`_imports` resolves both to the same module.
IMPURE_MODULES = frozenset({
    "os", "sys", "shutil", "subprocess", "socket", "select", "signal",
    "pathlib", "tempfile", "glob", "fileinput", "sqlite3", "shelve", "dbm",
    "random", "secrets", "uuid", "time", "datetime", "threading", "queue",
    "multiprocessing", "asyncio", "concurrent", "http", "urllib", "requests",
    "ftplib", "smtplib", "ssl", "webbrowser", "ctypes", "mmap", "gc", "atexit",
    "platform", "getpass", "resource", "importlib", "warnings", "traceback",
    "logging", "argparse", "sysconfig", "tkinter", "curses", "readline",
})

#: Methods that are a side effect on any receiver at all. ``replace`` is
#: deliberately absent: on a string it is the most common pure call there is,
#: and losing every function that formats text would empty this module out.
SIDE_EFFECTS = frozenset({
    "write", "writelines", "writerow", "writerows", "close", "seek", "flush",
    "truncate", "send", "sendall", "recv", "connect", "bind", "listen",
    "mkdir", "rmdir", "touch", "unlink", "chmod", "rename", "symlink_to",
    "write_text", "write_bytes", "read_text", "read_bytes", "resolve",
    "commit", "rollback", "execute", "executemany", "communicate", "kill",
    "terminate", "acquire", "release", "start", "join_thread", "set_result",
})

#: Methods that mutate what they are called on. They are subtracted from the
#: built-in table below, which is what makes ``items.append(x)`` on a
#: *parameter* a finding while the same call on a local list is not: building
#: a result with ``append`` is how most pure functions are written, appending
#: to the caller's list is a result the return value never mentions.
MUTATORS = frozenset({
    "append", "appendleft", "extend", "insert", "pop", "popleft", "popitem",
    "remove", "clear", "sort", "reverse", "update", "setdefault", "add",
    "discard", "difference_update", "intersection_update",
    "symmetric_difference_update",
})

#: The methods of the built-in types, taken from the interpreter rather than
#: written out here -- a hand-kept list of what ``str`` offers is a list that
#: is wrong by the next release. A call on a parameter is allowed only if it
#: names one of these, because everything else is a method of an object this
#: file does not describe: ``sandbox.read(path)`` and ``agent.ask(prompt)``
#: read exactly like a pure call and are neither.
BUILTIN_METHODS = frozenset(
    name for kind in (str, bytes, bytearray, list, tuple, dict, set,
                      frozenset, int, float, complex, bool)
    for name in dir(kind) if not name.startswith("_")
) - MUTATORS - SIDE_EFFECTS

#: Decorators a candidate may carry. Anything else can replace the function
#: outright -- a route, a fixture, a retry wrapper -- and what a property would
#: then be about is the decorator. ``cache`` is evidence rather than a problem:
#: memoising a function is only sound when it is already pure.
SAFE_DECORATORS = frozenset({"cache", "lru_cache", "cached", "final", "overload"})

#: Longest signature and docstring line carried into a prompt. Both are there
#: to pick a function out of a list, not to replace reading it.
SIGNATURE_CHARS = 110
DOC_CHARS = 90

#: Name pairs that promise an inverse. A round trip is the cheapest property
#: there is -- it needs no oracle and no invariant, only the two functions --
#: so it is worth naming for the Tester rather than leaving it to be noticed.
INVERSES = (
    ("encode", "decode"), ("serialise", "deserialise"),
    ("serialize", "deserialize"), ("dumps", "loads"), ("dump", "load"),
    ("pack", "unpack"), ("compress", "decompress"), ("deflate", "inflate"),
    ("marshal", "unmarshal"), ("escape", "unescape"), ("quote", "unquote"),
    ("wrap", "unwrap"), ("freeze", "thaw"), ("format", "parse"),
    ("render", "parse"), ("join", "split"), ("to", "from"),
)


@dataclass(frozen=True)
class Subject:
    """A function a property can quantify over, and how to read it in a list."""

    path: str
    name: str
    signature: str = ""
    doc: str = ""

    def render(self) -> str:
        line = f"{self.path}: {self.name}{self.signature}"
        return f"{line} -- {self.doc}" if self.doc else line


def subjects(sources: Iterable[tuple[str, str]], limit: int = 6) -> list[Subject]:
    """Property-testable functions in ``[(path, text)]``, in the order given.

    *sources* is what the prompt is already showing the agent, so a subject is
    never a function the model cannot see. Order is relevance order from
    :func:`hermes.implement.relevant_sources` and is kept rather than re-ranked:
    the first files are the ones the ticket names.

    *limit* is applied here rather than by the caller so that
    :func:`round_trips` pairs only functions that were actually offered; a
    round trip naming a partner that did not fit is worse than none.
    """
    found: list[Subject] = []
    for path, text in sources:
        if str(path).endswith(SUFFIXES):
            found += _module(str(path), text)
        if len(found) >= limit:
            break
    return found[:limit]


def round_trips(found: Sequence[Subject]) -> list[tuple[Subject, Subject]]:
    """Pairs among *found* whose names promise that one undoes the other."""
    pairs = []
    for index, first in enumerate(found):
        for second in found[index + 1:]:
            if _inverse(first.name, second.name):
                pairs.append((first, second))
    return pairs


def _module(path: str, text: str) -> list[Subject]:
    """Every function in one file that survives all of the rules above."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []       # a file that does not parse says nothing about purity
    worldly, imported = _imports(tree)
    # Purity is decided for every function in the file, including the private
    # helpers and the procedures that could never be offered: a public
    # function calling a private one is only as pure as what it called.
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    clean = {name for name, node in functions.items()
             if not _touches(node, worldly, imported)}
    # Anything else the file defines -- a class, an async def, a function
    # already rejected -- takes its callers down with it, and a chain of them
    # settles in at most one pass per function.
    defined = {node.name for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef))}
    for _ in range(len(functions)):
        shrunk = {name for name in clean
                  if not _calls(functions[name]) & (defined - clean)}
        if shrunk == clean:
            break
        clean = shrunk
    return [Subject(path, name, _signature(node), _docline(node))
            for name, node in functions.items()
            if name in clean and _shaped(node)]


def _shaped(node: ast.FunctionDef) -> bool:
    """Whether the function is even the right shape to quantify over.

    It needs an argument for the strategy to fill and a value to compare, so a
    nullary function and a procedure are both out. A generator is out for a
    subtler reason: its body has not run when it returns, and a property that
    forgets to drain it passes on code that raises.
    """
    if node.name.startswith("_"):
        return False    # a property pinned to a helper freezes what the next
                        # ticket is free to change
    args = node.args
    if not (args.posonlyargs or args.args or args.vararg or args.kwonlyargs):
        return False
    if any(_root(decorator) not in SAFE_DECORATORS for decorator in node.decorator_list):
        return False
    returns = False
    for child in _own(node):
        if isinstance(child, (ast.Yield, ast.YieldFrom)):
            return False
        if isinstance(child, ast.Return) and child.value is not None:
            returns = True
    return returns


def _touches(node: ast.FunctionDef, worldly: frozenset[str],
             imported: frozenset[str]) -> bool:
    """Whether anything in the body is visibly a side effect.

    *worldly* are the names this file bound to a module in
    :data:`IMPURE_MODULES`, so ``time.monotonic()`` and a ``from time import
    monotonic`` are the same finding here. *imported* is every name an import
    bound, which is how a module call is told from a method call.
    """
    params = _params(node)
    for child in _body(node):
        if isinstance(child, (ast.Global, ast.Nonlocal, ast.Await)):
            return True
        if isinstance(child, ast.Call) and _impure_call(child, params, worldly,
                                                        imported):
            return True
        if isinstance(child, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = (child.targets if isinstance(child, (ast.Assign, ast.Delete))
                       else [child.target])
            # Writing through a parameter is a result the return value does
            # not mention, and the second call in a property sees the first.
            if any(isinstance(t, (ast.Subscript, ast.Attribute))
                   and _root(t) in params for t in targets):
                return True
    return False


def _impure_call(call: ast.Call, params: frozenset[str], worldly: frozenset[str],
                 imported: frozenset[str]) -> bool:
    """Whether one call is visibly a side effect.

    The import check is what keeps ``json.dumps(payload)`` apart from
    ``sandbox.read(path)``: both are a dotted call on a name, and only the
    file's own import table says which of the two is a module.
    """
    func = call.func
    if isinstance(func, ast.Name):
        # A parameter that is called is an injected callable -- a runner, a
        # clock, a sleep -- and what it does is decided by the caller, which
        # is exactly what a property is not allowed to depend on.
        return (func.id in IMPURE_BUILTINS or func.id in worldly
                or func.id in params)
    if isinstance(func, ast.Attribute):
        if func.attr in SIDE_EFFECTS or _root(func) in worldly:
            return True
        if _root(func) in imported:
            return False
        return _root(func) in params and func.attr not in BUILTIN_METHODS
    return False


def _imports(tree: ast.Module) -> tuple[frozenset[str], frozenset[str]]:
    """The names this file's imports bound: the worldly ones, and all of them.

    Reading the import table beats guessing at bare names twice over. ``from
    os import getenv as env`` binds a name no list of suspicious words would
    hold, and the file itself says where it came from; and a name an import
    bound is a module or a function, never the method of an object, which is
    the distinction :func:`_impure_call` turns on.
    """
    worldly: set[str] = set()
    every: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                every.add(alias.asname or root)
                if root in IMPURE_MODULES:
                    worldly.add(alias.asname or root)
        elif isinstance(node, ast.ImportFrom):
            bound = {alias.asname or alias.name for alias in node.names}
            every |= bound
            if not node.level and (node.module or "").split(".")[0] in IMPURE_MODULES:
                worldly |= bound
    return frozenset(worldly), frozenset(every)


def _body(node: ast.AST):
    """Every node under *node*, including nested functions and lambdas.

    Descending into a closure is deliberate: a function that defines a writer
    and returns it has handed its caller the side effect rather than avoided
    it, and the rules here would rather lose that function than vouch for it.
    """
    for child in ast.walk(node):
        if child is not node:
            yield child


def _own(node: ast.AST):
    """Nodes in the function's own body, stopping at every function it nests.

    What makes a generator a generator, and what a ``return`` returns, are
    both properties of one function body; a ``yield`` inside a closure belongs
    to the closure and says nothing about the function around it.
    """
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        yield child
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.Lambda)):
            stack += list(ast.iter_child_nodes(child))


def _params(node: ast.FunctionDef) -> frozenset[str]:
    args = node.args
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    names += [a.arg for a in (args.vararg, args.kwarg) if a is not None]
    return frozenset(names)


def _calls(node: ast.FunctionDef) -> set[str]:
    """The leading name of everything the body calls, however it is spelled."""
    return {_root(child.func) for child in _body(node)
            if isinstance(child, ast.Call) and _root(child.func)}


def _root(node: ast.AST | None) -> str:
    """The leftmost name of a dotted expression: ``a`` of ``a.b.c()``."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def _inverse(first: str, second: str) -> bool:
    """Whether the two names are an inverse pair, as a whole or on one side.

    The shared remainder is what makes it a pair rather than a coincidence:
    ``encode_frame`` and ``decode_header`` are both in the table and undo
    nothing between them.
    """
    for left, right in INVERSES:
        for start, end in ((left, right), (right, left)):
            if first == start and second == end:
                return True
            if (first.startswith(f"{start}_") and second.startswith(f"{end}_")
                    and first[len(start):] == second[len(end):]):
                return True
            if (first.endswith(f"_{start}") and second.endswith(f"_{end}")
                    and first[:-len(start)] == second[:-len(end)]):
                return True
    return False


def _signature(node: ast.FunctionDef) -> str:
    """The argument list as written, or empty where it will not unparse."""
    try:
        rendered = " ".join(ast.unparse(node.args).split())
    except Exception:       # noqa: BLE001 - an unprintable annotation is not an error
        return "(…)"
    returns = ""
    if node.returns is not None:
        try:
            returns = f" -> {' '.join(ast.unparse(node.returns).split())}"
        except Exception:   # noqa: BLE001
            returns = ""
    return f"({rendered}){returns}"[:SIGNATURE_CHARS]


def _docline(node: ast.FunctionDef) -> str:
    doc = ast.get_docstring(node) or ""
    return doc.strip().split("\n", 1)[0].strip()[:DOC_CHARS]
