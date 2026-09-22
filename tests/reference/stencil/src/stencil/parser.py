import re

from .errors import TemplateSyntaxError
from .filters import FILTERS
from .lexer import tokenize

_TOKEN = re.compile(r"""\s*(?:(-?\d+)|('[^']*'|"[^"]*")|(==|!=|<=|>=|<|>|\||\(|\)|,|\.)|([A-Za-z_]\w*))""")


def _lex_expr(text, line):
    out, at = [], 0
    while at < len(text):
        if not text[at:].strip():
            break
        m = _TOKEN.match(text, at)
        if not m:
            raise TemplateSyntaxError(f"cannot read {text[at:].strip()!r}", line)
        number, string, op, name = m.groups()
        if number is not None:
            out.append(("lit", int(number)))
        elif string is not None:
            out.append(("lit", string[1:-1]))
        elif op is not None:
            out.append(("op", op))
        else:
            out.append(("name", name))
        at = m.end()
    return out


class _Expr:
    def __init__(self, text, line):
        self.tokens, self.at, self.line = _lex_expr(text, line), 0, line

    def fail(self, why):
        raise TemplateSyntaxError(why, self.line)

    def peek(self, kind=None, value=None):
        if self.at >= len(self.tokens):
            return None
        tok = self.tokens[self.at]
        if kind and tok[0] != kind or value is not None and tok[1] != value:
            return None
        return tok

    def take(self, kind=None, value=None):
        tok = self.peek(kind, value)
        if tok is None:
            self.fail(f"expected {value or kind or 'expression'}")
        self.at += 1
        return tok

    def parse(self):
        node = self.or_()
        if self.at != len(self.tokens):
            self.fail(f"unexpected {self.tokens[self.at][1]!r}")
        return node

    def or_(self):
        node = self.and_()
        while self.peek("name", "or"):
            self.at += 1
            node = ("or", node, self.and_())
        return node

    def and_(self):
        node = self.not_()
        while self.peek("name", "and"):
            self.at += 1
            node = ("and", node, self.not_())
        return node

    def not_(self):
        if self.peek("name", "not"):
            self.at += 1
            return ("not", self.not_())
        return self.compare()

    def compare(self):
        node = self.filtered()
        tok = self.peek("op")
        if tok and tok[1] in ("==", "!=", "<", ">", "<=", ">="):
            self.at += 1
            node = ("cmp", tok[1], node, self.filtered())
        return node

    def filtered(self):
        node = self.primary()
        while self.peek("op", "|"):
            self.at += 1
            name = self.take("name")[1]
            if name not in FILTERS:
                self.fail(f"unknown filter {name!r}")
            args = []
            if self.peek("op", "("):
                self.at += 1
                while not self.peek("op", ")"):
                    args.append(self.or_())
                    if not self.peek("op", ")"):
                        self.take("op", ",")
                self.take("op", ")")
            node = ("filter", name, node, args)
        return node

    def primary(self):
        tok = self.take()
        if tok[0] == "lit":
            return ("lit", tok[1])
        if tok == ("op", "("):
            node = self.or_()
            self.take("op", ")")
            return node
        if tok[0] == "name" and tok[1] not in ("and", "or", "not"):
            path = [tok[1]]
            while self.peek("op", "."):
                self.at += 1
                path.append(self.take("name")[1])
            return ("path", path)
        self.fail(f"unexpected {tok[1]!r}")


def expression(text, line):
    if not text:
        raise TemplateSyntaxError("empty expression", line)
    return _Expr(text, line).parse()


def parse(source):
    """A list of nodes: ("text", s) | ("out", expr) | ("if", branches, else)
    | ("for", name, expr, body, else)."""
    root = []
    stack = [("root", root, None, 0)]  # (kind, current body, block record, line)
    for kind, text, line in tokenize(source):
        body = stack[-1][1]
        if kind == "text":
            body.append(("text", text))
        elif kind == "output":
            body.append(("out", expression(text, line)))
        elif kind == "tag":
            word, _, rest = text.partition(" ")
            rest = rest.strip()
            top = stack[-1]
            if word == "if":
                block = {"kind": "if", "branches": [(expression(rest, line), [])],
                         "else": None}
                body.append(("if", block))
                stack.append(("if", block["branches"][0][1], block, line))
            elif word == "elif":
                if top[0] != "if" or top[2]["else"] is not None:
                    raise TemplateSyntaxError("elif outside if", line)
                branch = (expression(rest, line), [])
                top[2]["branches"].append(branch)
                stack[-1] = ("if", branch[1], top[2], top[3])
            elif word == "else":
                if top[0] not in ("if", "for") or top[2]["else"] is not None:
                    raise TemplateSyntaxError("else outside a block", line)
                top[2]["else"] = []
                stack[-1] = (top[0], top[2]["else"], top[2], top[3])
            elif word == "for":
                m = re.fullmatch(r"([A-Za-z_]\w*)\s+in\s+(.+)", rest)
                if not m:
                    raise TemplateSyntaxError("malformed for", line)
                block = {"kind": "for", "name": m.group(1),
                         "expr": expression(m.group(2), line), "body": [], "else": None}
                body.append(("for", block))
                stack.append(("for", block["body"], block, line))
            elif word in ("endif", "endfor"):
                if top[0] != word[3:]:
                    raise TemplateSyntaxError(f"{word} without opener", line)
                stack.pop()
            else:
                raise TemplateSyntaxError(f"unknown tag {word!r}", line)
    if len(stack) > 1:
        raise TemplateSyntaxError(f"{stack[-1][0]} never closed", stack[-1][3])
    return root
