"""A test runner for Hermes, written from the standard library alone.

Hermes ran its suite under pytest, and measured the factory's output by
installing pytest into the sandbox to run the held-out tests. Both are the same
dependency: a third-party package the factory cannot do its job without. This
module copies the part of pytest's behaviour Hermes uses -- not its code -- so
that the suite, the self-improvement gate and the held-out measurement run on a
bare interpreter.

What it copies:

* collection of ``test_*.py`` / ``*_test.py`` files, ``test*`` functions and
  the ``test*`` methods of ``Test*`` classes, and ``path::name[id]`` node ids;
* fixtures from the test module and the ``conftest.py`` files above it, at
  function, module or session scope, ``autouse`` included, with ``yield``
  fixtures torn down in reverse order of setup;
* the built-in fixtures ``tmp_path``, ``monkeypatch`` and ``capsys``;
* ``raises``, ``approx``, ``skip``, ``fail``, ``importorskip`` and the marks
  ``parametrize``, ``skip``, ``skipif`` and ``xfail``;
* output captured per test at the file-descriptor level, shown only for a
  failure; a JUnit XML report; ``-k``, ``-x``, ``-q``, ``-v``, ``-r``.

What it leaves out is everything Hermes never called: plugins, hooks, ini
files, assertion rewriting. A failed bare ``assert`` is explained from the
values of the names and side-effect-free expressions it mentions instead.

The module is self-contained on purpose. :mod:`hermes.harness` copies this one
file into a sandbox that has no Hermes installed, runs it as a script, and the
held-out tests there ``import testkit``; the factory's own tests use
``from hermes import testkit``. Either way the names resolve to the copy that
is running, because running it registers itself under both.

Run it as ``python -m hermes.testkit [options] [paths]``.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import importlib.util
import inspect
import io
import itertools
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import types
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable
from xml.sax.saxutils import escape, quoteattr

__all__ = ["approx", "fail", "fixture", "importorskip", "main", "mark", "param",
           "raises", "skip"]

_THIS = os.path.normcase(os.path.abspath(__file__)).rstrip("co")


# -- outcomes ------------------------------------------------------------------

class _Outcome(BaseException):
    """Raised to end a test early. ``BaseException`` so that the code under
    test cannot swallow it with ``except Exception``."""

    outcome = ""

    def __init__(self, reason: str = ""):
        super().__init__(reason)
        self.reason = reason


class Skipped(_Outcome):
    outcome = "skipped"


class Failed(_Outcome):
    outcome = "failed"


def _outcome_of(exc: BaseException) -> str:
    """The outcome an exception stands for, whichever copy of this module raised it."""
    return getattr(type(exc), "outcome", "") if isinstance(exc, BaseException) else ""


def skip(reason: str = "") -> None:
    """End the running test as skipped."""
    raise Skipped(reason)


def fail(reason: str = "") -> None:
    """End the running test as failed."""
    raise Failed(reason)


def importorskip(name: str, minversion: str | None = None, reason: str | None = None):
    """Import *name*, or skip the test when it is not installed."""
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        skip(reason or f"could not import {name!r}: {exc}")
    if minversion is not None:
        version = str(getattr(module, "__version__", "0"))
        if _version(version) < _version(minversion):
            skip(reason or f"module {name!r} has __version__ {version}, required is "
                           f">={minversion}")
    return module


def _version(text: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in re.split(r"[.+-]", text)[:4])


# -- raises --------------------------------------------------------------------

class ExceptionInfo:
    """What a :func:`raises` block caught."""

    def __init__(self):
        self.type: type | None = None
        self.value: BaseException | None = None
        self.tb = None

    @property
    def typename(self) -> str:
        return self.type.__name__ if self.type else ""

    def match(self, pattern: str) -> bool:
        if not re.search(pattern, str(self.value)):
            fail(f"Regex pattern {pattern!r} does not match {str(self.value)!r}.")
        return True

    def __repr__(self) -> str:
        return f"<ExceptionInfo {self.type.__name__ if self.type else None}: {self.value!r}>"


class _Raises:
    def __init__(self, expected, match: str | None):
        types_ = expected if isinstance(expected, tuple) else (expected,)
        if not types_ or not all(isinstance(t, type) and issubclass(t, BaseException)
                                 for t in types_):
            raise TypeError(f"expected exception must be a BaseException type, "
                            f"not {expected!r}")
        self.expected = types_
        self.pattern = match
        self.info = ExceptionInfo()

    def __enter__(self) -> ExceptionInfo:
        return self.info

    def __exit__(self, kind, value, tb) -> bool:
        if kind is None:
            names = " or ".join(t.__name__ for t in self.expected)
            fail(f"DID NOT RAISE {names}")
        if not issubclass(kind, self.expected):
            return False
        self.info.type, self.info.value, self.info.tb = kind, value, tb
        if self.pattern is not None and not re.search(self.pattern, str(value)):
            fail(f"Regex pattern did not match.\n Regex: {self.pattern!r}\n "
                 f"Input: {str(value)!r}")
        return True


def raises(expected, *args, match: str | None = None, **kwargs):
    """Assert that a block (``with raises(E):``) or a call raises *expected*."""
    context = _Raises(expected, match)
    if not args:
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {', '.join(kwargs)}")
        return context
    function, *rest = args
    with context:
        function(*rest, **kwargs)
    return context.info


# -- approx --------------------------------------------------------------------

class _Approx:
    def __init__(self, expected, rel: float | None, abs_: float | None):
        self.expected, self.rel, self.abs = expected, rel, abs_

    def _close(self, actual, expected) -> bool:
        if isinstance(expected, (list, tuple)):
            return (isinstance(actual, (list, tuple)) and len(actual) == len(expected)
                    and all(self._close(a, e) for a, e in zip(actual, expected)))
        if isinstance(expected, dict):
            return (isinstance(actual, dict) and actual.keys() == expected.keys()
                    and all(self._close(actual[k], expected[k]) for k in expected))
        if not isinstance(expected, (int, float)) or isinstance(expected, bool):
            return actual == expected
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False
        if math.isnan(expected) or math.isnan(actual):
            return False
        if math.isinf(expected) or math.isinf(actual):
            return actual == expected
        absolute = 1e-12 if self.abs is None else self.abs
        relative = (1e-6 if self.rel is None else self.rel) * abs(expected)
        if self.rel is None and self.abs is not None:
            relative = 0.0
        return abs(actual - expected) <= max(absolute, relative)

    def __eq__(self, actual) -> bool:
        return self._close(actual, self.expected)

    def __ne__(self, actual) -> bool:
        return not self == actual

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"approx({self.expected!r})"


def approx(expected, rel: float | None = None, abs: float | None = None):  # noqa: A002
    """A value equal to anything within a tolerance of *expected*."""
    return _Approx(expected, rel, abs)


# -- marks ---------------------------------------------------------------------

class Mark:
    def __init__(self, name: str, args: tuple = (), kwargs: dict | None = None):
        self.name, self.args, self.kwargs = name, args, dict(kwargs or {})

    def __repr__(self) -> str:
        return f"Mark({self.name!r}, {self.args!r}, {self.kwargs!r})"


class MarkDecorator:
    """``mark.name`` and ``mark.name(...)``: both decorate a test or a class."""

    def __init__(self, mark: Mark):
        self.mark = mark

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and (inspect.isfunction(args[0])
                                              or inspect.isclass(args[0])):
            target = args[0]
            marks = list(target.__dict__.get("testkit_marks", ()))
            marks.append(self.mark)
            target.testkit_marks = marks
            return target
        return MarkDecorator(Mark(self.mark.name, self.mark.args + args,
                                  {**self.mark.kwargs, **kwargs}))


class _MarkGenerator:
    def __getattr__(self, name: str) -> MarkDecorator:
        if name.startswith("_"):
            raise AttributeError(name)
        return MarkDecorator(Mark(name))


mark = _MarkGenerator()


class _Param:
    def __init__(self, values: tuple, id: str | None, marks):  # noqa: A002
        self.values, self.id = values, id
        self.marks = list(marks) if isinstance(marks, (list, tuple)) else [marks]


def param(*values, id: str | None = None, marks=()):  # noqa: A002
    """One parametrize case, with its own id or marks."""
    return _Param(values, id, marks)


def _marks_of(obj) -> list[Mark]:
    out = []
    for item in getattr(obj, "testkit_marks", ()) or ():
        out.append(item.mark if isinstance(item, MarkDecorator) else item)
    return out


# -- fixtures ------------------------------------------------------------------

class FixtureDef:
    def __init__(self, function: Callable, name: str, scope: str, autouse: bool):
        if scope not in ("function", "class", "module", "session", "package"):
            raise ValueError(f"fixture {name!r}: unknown scope {scope!r}")
        self.function, self.name, self.autouse = function, name, autouse
        self.scope = {"class": "module", "package": "session"}.get(scope, scope)
        self.params = [p for p in inspect.signature(function).parameters
                       if p not in ("self", "request")]


def fixture(function: Callable | None = None, *, scope: str = "function",
            autouse: bool = False, name: str | None = None, params=None, ids=None):
    """Declare a fixture. Usable bare (``@fixture``) or called (``@fixture(...)``)."""
    if params is not None:
        raise TypeError("parametrized fixtures are not supported; parametrize the test")

    def decorate(func: Callable) -> Callable:
        func.testkit_fixture = FixtureDef(func, name or func.__name__, scope, autouse)
        return func

    return decorate(function) if function is not None else decorate


def _fixtures_in(module) -> dict[str, FixtureDef]:
    out = {}
    for value in list(vars(module).values()):
        definition = getattr(value, "testkit_fixture", None)
        if definition is not None and hasattr(definition, "function"):
            out[definition.name] = definition
    return out


_NOTSET = object()


class MonkeyPatch:
    """Temporary changes to attributes, items, the environment and the cwd."""

    def __init__(self):
        self._undo: list[Callable[[], None]] = []

    @staticmethod
    def _resolve(dotted: str):
        parts = dotted.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            try:
                target = importlib.import_module(".".join(parts[:cut]))
            except ImportError:
                continue
            for attribute in parts[cut:-1]:
                target = getattr(target, attribute)
            return target, parts[-1]
        raise AttributeError(f"cannot import the module of {dotted!r}")

    def setattr(self, target, name=_NOTSET, value=_NOTSET, raising: bool = True) -> None:
        if value is _NOTSET:
            if not isinstance(target, str):
                raise TypeError("use setattr(target, name, value) or "
                                "setattr('dotted.path', value)")
            value = name
            target, name = self._resolve(target)
        old = (target.__dict__.get(name, _NOTSET) if inspect.isclass(target)
               else getattr(target, name, _NOTSET))
        if raising and old is _NOTSET and not hasattr(target, name):
            raise AttributeError(f"{target!r} has no attribute {name!r}")
        setattr(target, name, value)

        def undo():
            if old is _NOTSET:
                with contextlib.suppress(AttributeError):
                    delattr(target, name)
            else:
                setattr(target, name, old)
        self._undo.append(undo)

    def delattr(self, target, name=_NOTSET, raising: bool = True) -> None:
        if name is _NOTSET:
            target, name = self._resolve(target)
        if not hasattr(target, name):
            if raising:
                raise AttributeError(name)
            return
        old = (target.__dict__.get(name, _NOTSET) if inspect.isclass(target)
               else getattr(target, name))
        delattr(target, name)
        if old is not _NOTSET:
            self._undo.append(lambda: setattr(target, name, old))

    def setitem(self, mapping, key, value) -> None:
        old = mapping.get(key, _NOTSET) if hasattr(mapping, "get") else _NOTSET
        mapping[key] = value

        def undo():
            if old is _NOTSET:
                mapping.pop(key, None)
            else:
                mapping[key] = old
        self._undo.append(undo)

    def delitem(self, mapping, key, raising: bool = True) -> None:
        if key not in mapping:
            if raising:
                raise KeyError(key)
            return
        old = mapping.pop(key)
        self._undo.append(lambda: mapping.__setitem__(key, old))

    def setenv(self, name: str, value, prepend: str | None = None) -> None:
        value = str(value)
        old = os.environ.get(name)
        if prepend and old is not None:
            value = value + prepend + old
        os.environ[name] = value
        self._undo.append(lambda: _restore_env(name, old))

    def delenv(self, name: str, raising: bool = True) -> None:
        old = os.environ.get(name)
        if old is None:
            if raising:
                raise KeyError(name)
            return
        os.environ.pop(name)
        self._undo.append(lambda: _restore_env(name, old))

    def chdir(self, path) -> None:
        old = os.getcwd()
        os.chdir(path)
        self._undo.append(lambda: os.chdir(old))

    def syspath_prepend(self, path) -> None:
        old = list(sys.path)
        sys.path.insert(0, str(path))
        importlib.invalidate_caches()
        self._undo.append(lambda: sys.path.__setitem__(slice(None), old))

    def undo(self) -> None:
        while self._undo:
            self._undo.pop()()


def _restore_env(name: str, old: str | None) -> None:
    if old is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old


CaptureResult = namedtuple("CaptureResult", ["out", "err"])


class _Stream(io.TextIOWrapper):
    """A text stream that keeps what is written, the way sys.stdout is used."""

    def __init__(self):
        super().__init__(io.BytesIO(), encoding="utf-8", errors="replace",
                         newline="\n", write_through=True)

    def take(self) -> str:
        self.flush()
        value = self.buffer.getvalue().decode("utf-8", "replace")
        self.seek(0)
        self.truncate()
        return value


class CaptureFixture:
    """``capsys``: what the test printed to sys.stdout and sys.stderr so far."""

    def __init__(self, capture: "_Capture"):
        self._capture = capture

    def readouterr(self) -> CaptureResult:
        return CaptureResult(self._capture.out.take(), self._capture.err.take())


class _Capture:
    """Per-test capture of sys.stdout/err and, unless asked not to, of fds 1 and 2.

    The descriptors matter as much as the Python streams: a test that runs
    ``git`` or a server thread prints through them, and that output belongs
    with the test that caused it rather than in the middle of the progress line.
    """

    def __init__(self, mode: str):
        self.mode = mode
        self.out, self.err = _Stream(), _Stream()
        self._saved: list = []
        self.fd_text = ""

    def start(self) -> None:
        if self.mode == "no":
            return
        if self.mode == "fd":
            for fd in (1, 2):
                with contextlib.suppress(OSError):
                    sys.__stdout__ and sys.__stdout__.flush()
                    sys.__stderr__ and sys.__stderr__.flush()
                    spill = tempfile.TemporaryFile(buffering=0)
                    self._saved.append((fd, os.dup(fd), spill))
                    os.dup2(spill.fileno(), fd)
        self._streams = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = self.out, self.err

    def stop(self) -> tuple[str, str]:
        if self.mode == "no":
            return "", ""
        sys.stdout, sys.stderr = self._streams
        text = []
        for fd, saved, spill in reversed(self._saved):
            os.dup2(saved, fd)
            os.close(saved)
            spill.seek(0)
            text.append(spill.read().decode("utf-8", "replace"))
            spill.close()
        self._saved = []
        self.fd_text = "".join(reversed(text))
        return self.out.take(), self.err.take()


# -- collection ----------------------------------------------------------------

class Item:
    """One test: a function, maybe a method of a class, maybe one parameter set."""

    def __init__(self, path: Path, module, function: Callable, cls=None,
                 callspec: dict | None = None, case: str = "", marks=()):
        self.path, self.module, self.function, self.cls = path, module, function, cls
        self.callspec = callspec or {}
        self.case = case
        self.marks = list(marks)
        self.name = function.__name__ + (f"[{case}]" if case else "")
        self.relpath = _relative(path)
        self.nodeid = "::".join([self.relpath] + ([cls.__name__] if cls else []) + [self.name])

    @property
    def classname(self) -> str:
        """The JUnit ``classname``: the dotted module path, plus the class."""
        dotted = self.relpath[:-3].replace("/", ".") if self.relpath.endswith(".py") \
            else self.relpath.replace("/", ".")
        return dotted + (f".{self.cls.__name__}" if self.cls else "")

    def location(self) -> str:
        try:
            line = inspect.getsourcelines(self.function)[1]
        except (OSError, TypeError):
            line = 0
        return f"{self.relpath}:{line}"


def _relative(path: Path) -> str:
    try:
        return Path(os.path.relpath(path, os.getcwd())).as_posix()
    except ValueError:
        return Path(path).as_posix()


NORECURSE = {"build", "dist", "node_modules", "venv", "__pycache__", "CVS", "_darcs"}


def _is_test_file(path: Path) -> bool:
    return path.suffix == ".py" and (path.name.startswith("test_")
                                     or path.stem.endswith("_test"))


def _test_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    found = []
    for directory, subdirs, files in os.walk(target):
        subdirs[:] = sorted(d for d in subdirs if not d.startswith(".")
                            and d not in NORECURSE and not d.endswith(".egg-info")
                            and not d.endswith(".egg"))
        for name in sorted(files):
            if _is_test_file(Path(name)):
                found.append(Path(directory) / name)
    return found


def _basedir(path: Path) -> Path:
    """The first directory above *path* that is not a package: what goes on sys.path."""
    directory = path.parent
    while (directory / "__init__.py").is_file() and directory.parent != directory:
        directory = directory.parent
    return directory


def _module_name(path: Path, base: Path) -> str:
    relative = path.with_suffix("").relative_to(base)
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _import(path: Path):
    """Import a test or conftest file under the name its location gives it."""
    path = path.resolve()
    base = _basedir(path)
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    name = _module_name(path, base)
    existing = sys.modules.get(name)
    if existing is not None:
        source = getattr(existing, "__file__", None)
        if source and Path(source).resolve() == path:
            return existing
        # Two files with one name, as conftest.py in two directories without
        # __init__.py would be: give the second a name of its own.
        unique = f"{name}__testkit_{abs(hash(str(path))) % 10 ** 8}"
        spec = importlib.util.spec_from_file_location(unique, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(name)


def _ids_for(values, ids, argnames: list[str]) -> list[str]:
    if callable(ids):
        out = []
        for index, value in enumerate(values):
            given = ids(value)
            out.append(str(given) if given is not None else _auto_id(value, argnames, index))
        return out
    if ids is not None:
        out = [str(i) for i in ids]
        if len(out) != len(values):
            raise ValueError(f"{len(out)} ids for {len(values)} parameter sets")
        return out
    return [_auto_id(v, argnames, i) for i, v in enumerate(values)]


def _auto_id(value, argnames: list[str], index: int) -> str:
    parts = value if len(argnames) > 1 else (value,)
    pieces = []
    for name, part in zip(argnames, parts):
        if isinstance(part, (str, int, float, bool)) or part is None:
            text = str(part)
            if isinstance(part, str) and (not text.isprintable() or len(text) > 40):
                text = f"{name}{index}"
            pieces.append(text)
        else:
            pieces.append(f"{name}{index}")
    return "-".join(pieces)


def _expand(function: Callable, marks: list[Mark]) -> list[tuple[dict, str, list[Mark]]]:
    """Every parameter set the parametrize marks on *function* ask for."""
    cases: list[tuple[dict, str, list[Mark]]] = [({}, "", [])]
    for item in marks:
        if item.name != "parametrize":
            continue
        args = list(item.args)
        argnames = args[0] if args else item.kwargs.get("argnames")
        argvalues = args[1] if len(args) > 1 else item.kwargs.get("argvalues")
        if isinstance(argnames, str):
            argnames = [n.strip() for n in argnames.split(",") if n.strip()]
        argnames = list(argnames)
        values, case_marks, explicit = [], [], []
        for value in list(argvalues):
            if isinstance(value, _Param) or type(value).__name__ == "_Param":
                values.append(value.values if len(argnames) > 1 else value.values[0])
                case_marks.append([m.mark if isinstance(m, MarkDecorator) else m
                                   for m in value.marks])
                explicit.append(value.id)
            else:
                values.append(value)
                case_marks.append([])
                explicit.append(None)
        ids = _ids_for(values, item.kwargs.get("ids"), argnames)
        ids = [e if e is not None else i for e, i in zip(explicit, ids)]
        expanded = []
        for (spec, case, extra), (value, cid, more) in itertools.product(
                cases, zip(values, ids, case_marks)):
            row = dict(spec)
            parts = list(value) if len(argnames) > 1 else [value]
            if len(parts) != len(argnames):
                raise ValueError(f"{function.__name__}: {argnames} need "
                                 f"{len(argnames)} values, got {value!r}")
            row.update(zip(argnames, parts))
            expanded.append((row, f"{case}-{cid}" if case else cid, extra + more))
        cases = expanded
    return cases


def _collect_module(path: Path, module) -> list[Item]:
    items: list[Item] = []
    for name, obj in list(vars(module).items()):
        if inspect.isfunction(obj) and name.startswith("test"):
            marks = _marks_of(obj)
            for spec, case, extra in _expand(obj, marks):
                items.append(Item(path, module, obj, None, spec, case, marks + extra))
        elif inspect.isclass(obj) and name.startswith("Test"):
            if "__init__" in vars(obj) or getattr(obj, "__test__", True) is False:
                continue
            class_marks = _marks_of(obj)
            seen = set()
            for klass in obj.__mro__:
                if klass is object:
                    continue
                for attr, member in vars(klass).items():
                    if attr in seen or not attr.startswith("test"):
                        continue
                    seen.add(attr)
                    if not inspect.isfunction(member):
                        continue
                    marks = class_marks + _marks_of(member)
                    for spec, case, extra in _expand(member, marks):
                        items.append(Item(path, module, member, obj, spec, case,
                                          marks + extra))
    return items


# -- running -------------------------------------------------------------------

class Report:
    def __init__(self, item: Item | None, outcome: str, duration: float = 0.0,
                 longrepr: str = "", reason: str = "", where: str = "",
                 stdout: str = "", stderr: str = "", when: str = "call"):
        self.item, self.outcome, self.duration = item, outcome, duration
        self.longrepr, self.reason, self.where = longrepr, reason, where
        self.stdout, self.stderr, self.when = stdout, stderr, when


class _Scope:
    def __init__(self):
        self.values: dict[int, Any] = {}
        self.finalizers: list[tuple[str, Callable[[], None]]] = []

    def close(self) -> list[str]:
        """Run the teardowns, newest first. Returns the errors they raised."""
        errors = []
        while self.finalizers:
            name, finalize = self.finalizers.pop()
            try:
                finalize()
            except BaseException as exc:  # noqa: BLE001 - reported, not raised
                if isinstance(exc, KeyboardInterrupt):
                    raise
                errors.append(f"teardown of fixture {name!r}:\n" + _format(exc))
        self.values.clear()
        return errors


class Session:
    def __init__(self, options):
        self.options = options
        self.basetemp = Path(options.basetemp) if options.basetemp else Path(
            tempfile.mkdtemp(prefix="testkit-"))
        self.basetemp.mkdir(parents=True, exist_ok=True)
        self.scopes = {"session": _Scope(), "module": _Scope()}
        self.conftests: dict[Path, dict[str, FixtureDef]] = {}
        self.reports: list[Report] = []
        self.tmp_counter = itertools.count()
        self.capture: _Capture | None = None

    # -- fixtures ----------------------------------------------------------
    def _builtin(self) -> dict[str, FixtureDef]:
        def tmp_path(request_item):
            name = re.sub(r"[^\w.-]", "_", request_item.name)[:30]
            path = self.basetemp / f"{name}{next(self.tmp_counter)}"
            path.mkdir(parents=True)
            return path

        def monkeypatch():
            patch = MonkeyPatch()
            yield patch
            patch.undo()

        def capsys():
            return CaptureFixture(self.capture) if self.capture else None

        builtins = {}
        for function in (tmp_path, monkeypatch, capsys):
            definition = FixtureDef(function, function.__name__, "function", False)
            definition.params = []
            definition.wants_item = function is tmp_path
            builtins[function.__name__] = definition
        return builtins

    def _conftest_chain(self, path: Path) -> list[Path]:
        if self.options.noconftest:
            return []
        root = Path(os.getcwd()).resolve()
        chain = []
        directory = path.resolve().parent
        while True:
            candidate = directory / "conftest.py"
            if candidate.is_file():
                chain.append(candidate)
            if directory == root or directory.parent == directory:
                break
            if root not in directory.parents:
                break
            directory = directory.parent
        return list(reversed(chain))          # outermost first

    def visible(self, item: Item) -> dict[str, FixtureDef]:
        """Fixtures a test can ask for, nearest definition winning."""
        table = dict(self._builtins)
        for conftest in self._conftest_chain(item.path):
            table.update(self.conftests.get(conftest, {}))
        table.update(_fixtures_in(item.module))
        return table

    def load_conftests(self, path: Path) -> None:
        for conftest in self._conftest_chain(path):
            if conftest not in self.conftests:
                self.conftests[conftest] = _fixtures_in(_import(conftest))

    def resolve(self, name: str, item: Item, table: dict, function_scope: _Scope,
                stack: tuple = ()) -> Any:
        if name in item.callspec:
            return item.callspec[name]
        definition = table.get(name)
        if definition is None:
            raise _FixtureLookup(f"fixture {name!r} not found (available: "
                                 f"{', '.join(sorted(table))})")
        if name in stack:
            raise _FixtureLookup(f"fixture {name!r} depends on itself via "
                                 f"{' -> '.join(stack)}")
        scope = function_scope if definition.scope == "function" \
            else self.scopes[definition.scope]
        key = id(definition)
        if key in scope.values:
            return scope.values[key]
        kwargs = {p: self.resolve(p, item, table, function_scope, stack + (name,))
                  for p in definition.params}
        if getattr(definition, "wants_item", False):
            value = definition.function(item)
        else:
            value = definition.function(**kwargs)
        if inspect.isgenerator(value):
            generator = value
            value = next(generator)

            def finalize(generator=generator, name=name):
                try:
                    next(generator)
                except StopIteration:
                    return
                raise RuntimeError(f"fixture {name!r} yielded more than once")
            scope.finalizers.append((name, finalize))
        scope.values[key] = value
        return value

    # -- one test ----------------------------------------------------------
    def run(self, item: Item) -> Report:
        for definition in item.marks:
            if definition.name == "skip":
                reason = definition.kwargs.get("reason") or (
                    definition.args[0] if definition.args else "unconditional skip")
                return Report(item, "skipped", reason=reason, where=item.location())
            if definition.name == "skipif":
                condition = definition.args[0] if definition.args else \
                    definition.kwargs.get("condition", False)
                if condition:
                    return Report(item, "skipped",
                                  reason=definition.kwargs.get("reason", "condition"),
                                  where=item.location())
        xfail = next((m for m in item.marks if m.name == "xfail"), None)
        if xfail is not None and xfail.kwargs.get("run", True) is False:
            return Report(item, "xfailed", reason=xfail.kwargs.get("reason", ""))
        if xfail is not None and xfail.args and not xfail.args[0]:
            xfail = None                       # xfail(condition) that is false

        function_scope = _Scope()
        self.capture = _Capture(self.options.capture)
        self.capture.start()
        started = time.perf_counter()
        outcome, longrepr, reason, where, when = "passed", "", "", "", "call"
        try:
            table = self.visible(item)
            when = "setup"
            autouse = sorted((d for d in table.values() if d.autouse),
                             key=lambda d: ("session", "module", "function").index(d.scope))
            for definition in autouse:
                self.resolve(definition.name, item, table, function_scope)
            target = item.function
            if item.cls is not None:
                instance = item.cls()
                target = types.MethodType(item.function, instance)
                if hasattr(instance, "setup_method"):
                    instance.setup_method(item.function)
            signature = inspect.signature(target)
            kwargs = {}
            for name, parameter in signature.parameters.items():
                if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                    continue
                if (name not in item.callspec and name not in table
                        and parameter.default is not parameter.empty):
                    continue
                kwargs[name] = self.resolve(name, item, table, function_scope)
            when = "call"
            try:
                target(**kwargs)
            finally:
                if item.cls is not None and hasattr(target.__self__, "teardown_method"):
                    target.__self__.teardown_method(item.function)
        except KeyboardInterrupt:
            self.capture.stop()
            raise
        except _FixtureLookup as exc:
            outcome, longrepr = "error", f"{item.location()}: {exc}"
        except BaseException as exc:  # noqa: BLE001 - every exception is a result
            kind = _outcome_of(exc)
            if kind == "skipped":
                outcome, reason, where = "skipped", exc.reason, _raise_site(exc)
            elif when == "setup" and kind != "failed":
                outcome, longrepr = "error", _format(exc)
            else:
                outcome, longrepr = "failed", _format(exc)
                if kind == "failed":
                    longrepr = _format(exc, short=True)
        errors = function_scope.close()
        duration = time.perf_counter() - started
        stdout, stderr = self.capture.stop()
        stdout = self.capture.fd_text + stdout if self.capture.fd_text else stdout
        self.capture = None
        if errors and outcome in ("passed", "skipped"):
            outcome, longrepr = "error", "\n".join(errors)
        elif errors:
            longrepr += "\n" + "\n".join(errors)
        if xfail is not None:
            if outcome == "failed":
                outcome, reason = "xfailed", xfail.kwargs.get("reason", "")
            elif outcome == "passed":
                if xfail.kwargs.get("strict"):
                    outcome, longrepr = "failed", "[XPASS(strict)] " + xfail.kwargs.get(
                        "reason", "")
                else:
                    outcome = "xpassed"
        return Report(item, outcome, duration, longrepr, reason, where, stdout, stderr,
                      when)

    def end_module(self) -> list[str]:
        return self.scopes["module"].close()

    def end_session(self) -> list[str]:
        errors = self.scopes["module"].close() + self.scopes["session"].close()
        if not self.options.basetemp:
            _remove(self.basetemp)
        return errors


class _FixtureLookup(Exception):
    pass


def _remove(path: Path) -> None:
    def unlock(function, target, _info):
        with contextlib.suppress(OSError):
            os.chmod(target, 0o700)
            parent = os.path.dirname(target)
            os.chmod(parent, 0o700)
            function(target)
    shutil.rmtree(path, onerror=unlock)


# -- reporting -----------------------------------------------------------------

def _ours(filename: str) -> bool:
    name = os.path.normcase(os.path.abspath(filename))
    return name.rstrip("co") == _THIS or os.path.basename(name) in ("testkit.py",)


def _raise_site(exc: BaseException) -> str:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if not _ours(f.filename)]
    if not frames:
        return ""
    frame = frames[-1]
    return f"{_relative(Path(frame.filename))}:{frame.lineno}"


def _format(exc: BaseException, short: bool = False) -> str:
    """A traceback with this module's frames left out, and a bare assert explained."""
    if short:
        return f"{_raise_site(exc)}: {getattr(exc, 'reason', '') or exc}"
    tb = exc.__traceback__
    entries = traceback.extract_tb(tb)
    kept = [e for e in entries if not _ours(e.filename)] or entries
    lines = ["Traceback (most recent call last):\n"] + traceback.format_list(kept)
    lines += traceback.format_exception_only(type(exc), exc)
    text = "".join(lines)
    if isinstance(exc, AssertionError):
        explained = _explain(tb)
        if explained:
            text += explained
    return text.rstrip()


def _explain(tb) -> str:
    """The values a failed ``assert`` compared, read without calling anything."""
    frame = None
    while tb is not None:
        if not _ours(tb.tb_frame.f_code.co_filename):
            frame, lineno = tb.tb_frame, tb.tb_lineno
        tb = tb.tb_next
    if frame is None:
        return ""
    try:
        source = inspect.getsource(frame.f_code) if frame.f_code.co_name != "<module>" \
            else Path(frame.f_code.co_filename).read_text()
        first = frame.f_code.co_firstlineno if frame.f_code.co_name != "<module>" else 1
        tree = ast.parse(_dedent(source))
    except (OSError, TypeError, SyntaxError):
        return ""
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            start = node.lineno + first - 1
            end = (node.end_lineno or node.lineno) + first - 1
            if start <= lineno <= end:
                target = node
    if target is None:
        return ""
    shown, seen = [], set()
    impure = (ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr, ast.Lambda,
              ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

    def show(node) -> None:
        # A comparison or a boolean is explained by its operands, which are
        # what the reader needs; the whole is already known to be false.
        if isinstance(node, ast.Compare):
            for part in (node.left, *node.comparators):
                show(part)
            return
        if isinstance(node, ast.BoolOp):
            for part in node.values:
                show(part)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            show(node.operand)
            return
        if isinstance(node, ast.Constant):
            return
        if any(isinstance(n, impure) for n in ast.walk(node)):
            # Evaluating it again could act twice; show what it was made of.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    show(child)
            return
        text = ast.unparse(node)
        if text in seen:
            return
        seen.add(text)
        try:
            value = eval(compile(ast.Expression(node), "<assert>", "eval"),  # noqa: S307
                         frame.f_globals, frame.f_locals)
        except Exception:  # noqa: BLE001 - an explanation is best effort
            return
        if callable(value) and not isinstance(value, type):
            if isinstance(node, ast.Attribute):
                show(node.value)         # a method: what it was called on
            return
        rendered = repr(value)
        if len(rendered) > 600:
            rendered = rendered[:600] + "..."
        shown.append(f"    {text} = {rendered}")

    show(target.test)
    if not shown:
        return ""
    return "\n  where\n" + "\n".join(shown)


def _dedent(source: str) -> str:
    lines = source.splitlines()
    indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join(l[indent:] for l in lines)


SYMBOLS = {"passed": ".", "failed": "F", "error": "E", "skipped": "s",
           "xfailed": "x", "xpassed": "X"}


def _summary(counts: dict[str, int], elapsed: float) -> str:
    order = ("failed", "passed", "skipped", "xfailed", "xpassed", "error")
    names = {"error": "errors"}
    parts = []
    for key in order:
        if counts.get(key):
            label = names.get(key, key) if counts[key] != 1 or key != "error" else "error"
            parts.append(f"{counts[key]} {label}")
    return f"{', '.join(parts) or 'no tests ran'} in {elapsed:.2f}s"


def _junit(path: str, reports: list[Report], elapsed: float) -> None:
    counts = {k: sum(1 for r in reports if r.outcome == k)
              for k in ("failed", "error", "skipped")}
    lines = ['<?xml version="1.0" encoding="utf-8"?>',
             f'<testsuites><testsuite name="testkit" errors="{counts["error"]}" '
             f'failures="{counts["failed"]}" skipped="{counts["skipped"]}" '
             f'tests="{len(reports)}" time="{elapsed:.3f}">']
    for report in reports:
        if report.item is None:
            classname, name = report.where.replace("/", ".").rsplit(".py", 1)[0], report.where
        else:
            classname, name = report.item.classname, report.item.name
        head = f'<testcase classname={quoteattr(classname)} name={quoteattr(name)} ' \
               f'time="{report.duration:.3f}"'
        body = ""
        if report.outcome == "failed":
            body = (f'<failure message={quoteattr(report.longrepr.splitlines()[-1][:200] if report.longrepr else "failed")}>'
                    f"{escape(_xml_safe(report.longrepr))}</failure>")
        elif report.outcome == "error":
            body = (f'<error message={quoteattr("error")}>'
                    f"{escape(_xml_safe(report.longrepr))}</error>")
        elif report.outcome in ("skipped", "xfailed"):
            body = f'<skipped type="testkit.skip" message={quoteattr(_xml_safe(report.reason))}/>'
        if report.stdout and report.outcome in ("failed", "error"):
            body += f"<system-out>{escape(_xml_safe(report.stdout))}</system-out>"
        lines.append(head + (f">{body}</testcase>" if body else "/>"))
    lines.append("</testsuite></testsuites>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _xml_safe(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]", "?", text or "")


# -- selection -----------------------------------------------------------------

def _keyword_filter(expression: str) -> Callable[[Item], bool]:
    """``-k``: words joined by and/or/not, each a case-insensitive substring match."""
    tokens = re.findall(r"\(|\)|[^\s()]+", expression)
    code = []
    for token in tokens:
        if token in ("and", "or", "not", "(", ")"):
            code.append(token)
        else:
            code.append(f"_m({token!r})")
    compiled = compile(" ".join(code) or "True", "<-k>", "eval")

    def accept(item: Item) -> bool:
        haystack = " ".join([item.nodeid, *(m.name for m in item.marks)]).lower()
        return bool(eval(compiled, {"__builtins__": {}},  # noqa: S307 - built above
                         {"_m": lambda word: word.lower() in haystack}))
    return accept


def _split_nodeid(arg: str) -> tuple[str, list[str]]:
    path, *names = arg.split("::")
    return path, names


# -- entry point ---------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="testkit", description="Run tests without third-party software.")
    parser.add_argument("paths", nargs="*", help="files, directories or node ids")
    parser.add_argument("-q", "--quiet", action="count", default=0)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-x", "--exitfirst", action="store_true")
    parser.add_argument("-k", dest="keyword", default="", help="select by expression")
    parser.add_argument("-r", dest="report", default="",
                        help="summarise these outcomes: s skipped, x xfailed, "
                             "f failed, E error, a all but passed")
    parser.add_argument("-s", dest="capture", action="store_const", const="no",
                        help="do not capture output")
    parser.add_argument("--capture", choices=("fd", "sys", "no"), default="fd")
    parser.add_argument("--junitxml", help="write a JUnit XML report here")
    parser.add_argument("--noconftest", action="store_true",
                        help="do not load conftest.py files")
    parser.add_argument("--basetemp", help="put tmp_path directories here, and keep them")
    parser.add_argument("--collect-only", "--co", dest="collect_only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the tests. Exit 0 on success, 1 on failures, 2 on interruption or a
    usage error, 5 when nothing was collected -- pytest's codes, so callers
    that read them need not learn new ones."""
    options = _parser().parse_args(argv)
    if os.getcwd() not in sys.path and "" not in sys.path:
        sys.path.insert(0, os.getcwd())     # as ``python -m`` would
    targets = options.paths or (["tests"] if Path("tests").is_dir() else ["."])
    session = Session(options)
    session._builtins = session._builtin()
    started = time.perf_counter()
    write = sys.stdout.write
    items: list[Item] = []
    collection_errors: list[Report] = []
    for target in targets:
        path_text, names = _split_nodeid(target)
        path = Path(path_text)
        if not path.exists():
            print(f"ERROR: file or directory not found: {target}", file=sys.stderr)
            return 4
        for test_file in _test_files(path):
            try:
                session.load_conftests(test_file)
                module = _import(test_file)
                collected = _collect_module(test_file, module)
            except KeyboardInterrupt:
                raise
            except BaseException as exc:  # noqa: BLE001 - reported as an error
                if _outcome_of(exc) == "skipped":
                    collection_errors.append(Report(None, "skipped", reason=exc.reason,
                                                    where=_relative(test_file)))
                    continue
                collection_errors.append(Report(None, "error", longrepr=_format(exc),
                                                where=_relative(test_file), when="collect"))
                continue
            if names:
                wanted = names[-1]
                collected = [i for i in collected
                             if (i.name == wanted or i.function.__name__ == wanted)
                             and (len(names) < 2 or (i.cls and i.cls.__name__ == names[0]))]
            items.extend(collected)
    if options.keyword:
        keep = _keyword_filter(options.keyword)
        items = [i for i in items if keep(i)]

    if options.collect_only:
        for item in items:
            print(item.nodeid)
        for report in collection_errors:
            print(f"ERROR collecting {report.where}\n{report.longrepr}")
        return 0 if items and not collection_errors else (2 if collection_errors else 5)

    verbose = options.verbose - options.quiet
    reports: list[Report] = list(collection_errors)
    column = 0
    interrupted = False
    current_module = None
    try:
        for item in items:
            if current_module is not None and item.module is not current_module:
                for error in session.end_module():
                    reports.append(Report(None, "error", longrepr=error, where=item.relpath))
            current_module = item.module
            report = session.run(item)
            reports.append(report)
            if verbose > 0:
                write(f"{item.nodeid} {report.outcome.upper()}\n")
            else:
                write(SYMBOLS[report.outcome])
                column += 1
                if column % 72 == 0:
                    write(f" [{100 * len([r for r in reports if r.item]) // max(1, len(items)):3d}%]\n")
            sys.stdout.flush()
            if options.exitfirst and report.outcome in ("failed", "error"):
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        for error in session.end_session():
            reports.append(Report(None, "error", longrepr=error, where="session"))
    if verbose <= 0 and column % 72:
        write("\n")
    elapsed = time.perf_counter() - started

    bad = [r for r in reports if r.outcome in ("failed", "error")]
    if bad:
        write("\n" + "=" * 30 + " FAILURES " + "=" * 30 + "\n")
        for report in bad:
            title = report.item.nodeid if report.item else report.where
            write(f"{'_' * 10} {title} {'_' * 10}\n{report.longrepr}\n")
            if report.stdout.strip():
                write(f"{'-' * 10} captured stdout {'-' * 10}\n{report.stdout.rstrip()}\n")
            if report.stderr.strip():
                write(f"{'-' * 10} captured stderr {'-' * 10}\n{report.stderr.rstrip()}\n")
    wants = set(options.report.replace("a", "sxXfE"))
    lines = []
    if "s" in wants:
        grouped: dict[tuple[str, str], int] = {}
        for report in reports:
            if report.outcome == "skipped":
                key = (report.where or (report.item.location() if report.item else ""),
                       report.reason)
                grouped[key] = grouped.get(key, 0) + 1
        lines += [f"SKIPPED [{n}] {where}: {reason}" for (where, reason), n in grouped.items()]
    if "x" in wants:
        lines += [f"XFAIL {r.item.nodeid} - {r.reason}" for r in reports
                  if r.outcome == "xfailed" and r.item]
    if "X" in wants:
        lines += [f"XPASS {r.item.nodeid}" for r in reports if r.outcome == "xpassed"]
    lines += [f"{'FAILED' if r.outcome == 'failed' else 'ERROR'} "
              f"{r.item.nodeid if r.item else r.where}" for r in bad]
    if lines:
        write("=" * 24 + " short test summary info " + "=" * 24 + "\n")
        write("\n".join(lines) + "\n")
    counts: dict[str, int] = {}
    for report in reports:
        counts[report.outcome] = counts.get(report.outcome, 0) + 1
    if interrupted:
        write("!!! interrupted !!!\n")
    write(_summary(counts, elapsed) + "\n")
    sys.stdout.flush()
    if options.junitxml:
        _junit(options.junitxml, reports, elapsed)
    if interrupted:
        return 2
    if bad:
        return 1
    if not items:
        return 5
    return 0


def _register_self() -> None:
    """Make every name this file can be imported under resolve to this copy.

    Run as a script or with ``-m``, the module is ``__main__``; a test that
    then imports ``testkit`` or ``hermes.testkit`` would otherwise load a
    second copy, whose marks and fixtures this one would still read but whose
    module state it would not share.
    """
    me = sys.modules[__name__]
    spec = getattr(me, "__spec__", None)
    names = {"testkit"}
    if spec is not None and spec.name:
        names.add(spec.name)
    for name in names:
        sys.modules.setdefault(name, me)
        parent, _, child = name.rpartition(".")
        if parent and parent in sys.modules:
            setattr(sys.modules[parent], child, me)


if __name__ == "__main__":
    _register_self()
    sys.exit(main())
