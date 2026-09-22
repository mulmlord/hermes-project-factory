from collections import ChainMap
from collections.abc import Mapping
from types import SimpleNamespace

from .errors import UndefinedError
from .filters import FILTERS, Markup, escape
from .parser import parse


class _Undefined:
    def __init__(self, name):
        self.name = name


def _lookup(value, key):
    if isinstance(value, Mapping) and key in value:
        return value[key]
    if hasattr(value, key):
        return getattr(value, key)
    return _Undefined(key)


def _defined(value, what="value"):
    if isinstance(value, _Undefined):
        raise UndefinedError(f"{value.name!r} is undefined")
    return value


def _eval(node, ctx):
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "path":
        head, *rest = node[1]
        value = ctx[head] if head in ctx else _Undefined(head)
        for key in rest:
            value = value if isinstance(value, _Undefined) else _lookup(value, key)
        return value
    if kind == "filter":
        _, name, inner, args = node
        value = _eval(inner, ctx)
        argv = [_defined(_eval(a, ctx)) for a in args]
        if name == "default":
            return argv[0] if isinstance(value, _Undefined) or value is None else value
        return FILTERS[name](_defined(value), *argv)
    if kind == "not":
        return not _truth(_eval(node[1], ctx))
    if kind == "and":
        return _truth(_eval(node[1], ctx)) and _truth(_eval(node[2], ctx))
    if kind == "or":
        return _truth(_eval(node[1], ctx)) or _truth(_eval(node[2], ctx))
    if kind == "cmp":
        _, op, a, b = node
        a, b = _defined(_eval(a, ctx)), _defined(_eval(b, ctx))
        return {"==": a == b, "!=": a != b}.get(op) if op in ("==", "!=") else {
            "<": lambda: a < b, ">": lambda: a > b,
            "<=": lambda: a <= b, ">=": lambda: a >= b}[op]()
    raise AssertionError(kind)


def _truth(value):
    return False if isinstance(value, _Undefined) else bool(value)


def _render(nodes, ctx, autoescape, out):
    for node in nodes:
        kind = node[0]
        if kind == "text":
            out.append(node[1])
        elif kind == "out":
            value = _defined(_eval(node[1], ctx))
            if value is None:
                continue
            out.append(escape(value) if autoescape else str(value))
        elif kind == "if":
            block = node[1]
            for cond, body in block["branches"]:
                if _truth(_eval(cond, ctx)):
                    _render(body, ctx, autoescape, out)
                    break
            else:
                _render(block["else"] or [], ctx, autoescape, out)
        elif kind == "for":
            block = node[1]
            items = list(_defined(_eval(block["expr"], ctx)))
            for i, item in enumerate(items):
                loop = SimpleNamespace(index=i + 1, first=i == 0, last=i == len(items) - 1)
                _render(block["body"], ChainMap({block["name"]: item, "loop": loop}, ctx),
                        autoescape, out)
            if not items:
                _render(block["else"] or [], ctx, autoescape, out)


class Template:
    def __init__(self, source, autoescape=False):
        self.nodes = parse(source)
        self.autoescape = autoescape

    def render(self, context=None, **names):
        out = []
        _render(self.nodes, ChainMap(dict(names), dict(context or {})), self.autoescape, out)
        return "".join(out)


def render(source, context=None, **names):
    return Template(source).render(context, **names)
