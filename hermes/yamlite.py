"""The YAML Hermes reads and writes, without a YAML library.

Hermes used PyYAML for one file: its own configuration. That file is block
mappings, a few block sequences, flow collections such as
``{candidates: 2, files: 10}``, scalars and comments. This module reads and
writes exactly that subset, from the standard library alone, so the factory has
no third-party package to install before it can read its settings.

Scalars resolve the way PyYAML's ``safe_load`` resolves them -- YAML 1.1, not
1.2 -- because configurations written before this module existed must keep
meaning what they meant. ``speculative: off`` is still ``False``, ``0755`` is
still octal, and ``1e3`` is still the string it always was under PyYAML.

What is outside the subset is refused by name rather than guessed at: anchors,
aliases, tags, directives and multi-document streams raise :class:`YAMLError`.
A configuration that needs them is not one Hermes writes, and reading half of
one would hand the factory settings nobody chose.
"""
from __future__ import annotations

import math
import re
from typing import Any

__all__ = ["YAMLError", "load", "dump", "flow"]


class YAMLError(ValueError):
    """The text is not YAML this module reads. Carries the 1-based line."""

    def __init__(self, message: str, line: int | None = None):
        self.line = line
        super().__init__(f"line {line}: {message}" if line else message)


# -- scalars -------------------------------------------------------------------

_NULL = {"", "~", "null", "Null", "NULL"}
_TRUE = {"yes", "Yes", "YES", "true", "True", "TRUE", "on", "On", "ON"}
_FALSE = {"no", "No", "NO", "false", "False", "FALSE", "off", "Off", "OFF"}
_INT = re.compile(r"""^(?:[-+]?0b[01_]+
                       |[-+]?0[0-7_]+
                       |[-+]?(?:0|[1-9][0-9_]*)
                       |[-+]?0x[0-9a-fA-F_]+)$""", re.X)
_FLOAT = re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
                         |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
                         |[-+]?\.(?:inf|Inf|INF)
                         |\.(?:nan|NaN|NAN))$""", re.X)


def resolve(text: str) -> Any:
    """The value a plain (unquoted) scalar stands for, as PyYAML reads it."""
    if text in _NULL:
        return None
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    if _INT.match(text):
        digits = text.replace("_", "")
        sign = -1 if digits.startswith("-") else 1
        body = digits.lstrip("+-")
        if body.startswith("0b"):
            return sign * int(body[2:], 2)
        if body.startswith("0x"):
            return sign * int(body[2:], 16)
        if len(body) > 1 and body.startswith("0"):
            return sign * int(body, 8)
        return sign * int(body)
    if _FLOAT.match(text):
        lowered = text.replace("_", "").lower()
        if lowered.endswith(".inf"):
            return -math.inf if lowered.startswith("-") else math.inf
        if lowered == ".nan":
            return math.nan
        return float(lowered)
    return text


_ESCAPES = {"0": "\0", "a": "\a", "b": "\b", "t": "\t", "\t": "\t", "n": "\n",
            "v": "\v", "f": "\f", "r": "\r", "e": "\x1b", " ": " ", '"': '"',
            "/": "/", "\\": "\\", "N": "\x85", "_": "\xa0", "L": "\u2028",
            "P": "\u2029"}
_HEX = {"x": 2, "u": 4, "U": 8}


def _double_quoted(body: str, line: int) -> str:
    out, i = [], 0
    while i < len(body):
        char = body[i]
        if char != "\\":
            out.append(char)
            i += 1
            continue
        if i + 1 >= len(body):
            raise YAMLError("a double-quoted string ends in a backslash", line)
        code = body[i + 1]
        if code in _ESCAPES:
            out.append(_ESCAPES[code])
            i += 2
        elif code in _HEX:
            width = _HEX[code]
            digits = body[i + 2:i + 2 + width]
            if len(digits) != width or not all(c in "0123456789abcdefABCDEF" for c in digits):
                raise YAMLError(f"bad \\{code} escape in a double-quoted string", line)
            out.append(chr(int(digits, 16)))
            i += 2 + width
        else:
            raise YAMLError(f"unknown escape \\{code} in a double-quoted string", line)
    return "".join(out)


# -- reading -------------------------------------------------------------------

class _Text:
    """A cursor over one string: the flow parser's whole state."""

    def __init__(self, text: str, line: int):
        self.text, self.pos, self.line = text, 0, line

    def peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def skip_space(self) -> None:
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char in " \t\n":
                self.pos += 1
            elif char == "#" and (self.pos == 0 or self.text[self.pos - 1] in " \t\n"):
                end = self.text.find("\n", self.pos)
                self.pos = len(self.text) if end < 0 else end
            else:
                break

    def fail(self, message: str) -> YAMLError:
        return YAMLError(message, self.line)


def _quoted(cursor: _Text) -> str:
    """Read a quoted scalar starting at the cursor; leave it just past it."""
    quote = cursor.peek()
    cursor.pos += 1
    out = []
    while True:
        if cursor.pos >= len(cursor.text):
            raise cursor.fail(f"an unterminated {quote}-quoted string")
        char = cursor.text[cursor.pos]
        if quote == "'" and char == "'":
            if cursor.text[cursor.pos + 1:cursor.pos + 2] == "'":
                out.append("'")
                cursor.pos += 2
                continue
            cursor.pos += 1
            break
        if quote == '"' and char == "\\":
            out.append(cursor.text[cursor.pos:cursor.pos + 2])
            cursor.pos += 2
            continue
        if quote == '"' and char == '"':
            cursor.pos += 1
            break
        out.append(char)
        cursor.pos += 1
    body = "".join(out)
    # A quoted scalar broken over lines folds each break into one space.
    body = re.sub(r"[ \t]*\n[ \t]*", " ", body)
    return _double_quoted(body, cursor.line) if quote == '"' else body


def _flow_value(cursor: _Text) -> Any:
    cursor.skip_space()
    char = cursor.peek()
    if not char:
        raise cursor.fail("an unterminated flow collection")
    if char == "[":
        return _flow_sequence(cursor)
    if char == "{":
        return _flow_mapping(cursor)
    if char in ("'", '"'):
        return _quoted(cursor)
    if char in "&*!|>%@`":
        raise cursor.fail(f"{char!r} is outside the YAML Hermes reads")
    start = cursor.pos
    while cursor.pos < len(cursor.text):
        char = cursor.text[cursor.pos]
        if char in ",[]{}":
            break
        if char == ":" and cursor.text[cursor.pos + 1:cursor.pos + 2] in ("", " ", "\t",
                                                                           "\n", ",", "}", "]"):
            break
        if char == "#" and cursor.text[cursor.pos - 1:cursor.pos] in (" ", "\t", "\n"):
            break
        cursor.pos += 1
    # Only a line break folds, into one space; spaces inside a line are kept.
    text = cursor.text[start:cursor.pos].strip(" \t\n")
    return resolve(re.sub(r"[ \t]*\n[ \t]*", " ", text))


def _flow_sequence(cursor: _Text) -> list:
    cursor.pos += 1
    items: list = []
    while True:
        cursor.skip_space()
        if cursor.peek() == "]":
            cursor.pos += 1
            return items
        if not cursor.peek():
            raise cursor.fail("an unterminated [ sequence")
        items.append(_flow_value(cursor))
        cursor.skip_space()
        if not cursor.peek():
            raise cursor.fail("an unterminated [ sequence")
        if cursor.peek() == ",":
            cursor.pos += 1
        elif cursor.peek() != "]":
            raise cursor.fail(f"expected , or ] in a flow sequence, found {cursor.peek()!r}")


def _flow_mapping(cursor: _Text) -> dict:
    cursor.pos += 1
    out: dict = {}
    while True:
        cursor.skip_space()
        if cursor.peek() == "}":
            cursor.pos += 1
            return out
        if not cursor.peek():
            raise cursor.fail("an unterminated { mapping")
        key = _flow_value(cursor)
        cursor.skip_space()
        value = None
        if cursor.peek() == ":":
            cursor.pos += 1
            cursor.skip_space()
            if cursor.peek() not in (",", "}"):
                value = _flow_value(cursor)
        _hashable(key, cursor.line)
        out[key] = value
        cursor.skip_space()
        if not cursor.peek():
            raise cursor.fail("an unterminated { mapping")
        if cursor.peek() == ",":
            cursor.pos += 1
        elif cursor.peek() != "}":
            raise cursor.fail(f"expected , or }} in a flow mapping, found {cursor.peek()!r}")


def _hashable(key: Any, line: int) -> None:
    if isinstance(key, (list, dict)):
        raise YAMLError("a mapping key must be a scalar", line)


def _strip_comment(text: str) -> str:
    """*text* without a trailing comment, honouring quotes."""
    quote = ""
    for i, char in enumerate(text):
        if quote:
            if char == quote:
                if quote == "'" and text[i + 1:i + 2] == "'":
                    continue
                quote = ""
            elif char == "\\" and quote == '"':
                continue
        elif char in ("'", '"') and (i == 0 or text[i - 1] in " \t[{,:-"):
            quote = char
        elif char == "#" and (i == 0 or text[i - 1] in " \t"):
            return text[:i].rstrip()
    return text.rstrip()


def _split_key(content: str, line: int) -> tuple[Any, int] | None:
    """``(key, offset)`` when *content* opens a mapping entry, else ``None``.

    *offset* is where the value begins in *content*, so the caller can read
    the value from the original line with its comment still attached.
    """
    if content[:1] in ("'", '"'):
        cursor = _Text(content, line)
        try:
            key = _quoted(cursor)
        except YAMLError:
            return None
        rest = content[cursor.pos:]
        if rest.startswith(":") and rest[1:2] in ("", " ", "\t"):
            return key, cursor.pos + 1
        return None
    if content[:1] in "[{":
        return None
    for i, char in enumerate(content):
        if char == "#" and (i == 0 or content[i - 1] in " \t"):
            return None
        if char == ":" and content[i + 1:i + 2] in ("", " ", "\t"):
            return resolve(content[:i].strip()), i + 1
    return None


def _indent(raw: str, line: int) -> int:
    stripped = raw.lstrip(" ")
    if stripped.startswith("\t"):
        raise YAMLError("tabs may not indent YAML", line)
    return len(raw) - len(stripped)


def _significant(raw: str) -> bool:
    content = raw.strip()
    return bool(content) and not content.startswith("#")


class _Reader:
    def __init__(self, text: str):
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if lines and lines[0].startswith("\ufeff"):
            lines[0] = lines[0][1:]
        body = [raw for raw in lines if _significant(raw)]
        if body and body[0].startswith("%"):
            raise YAMLError("directives are outside the YAML Hermes reads", 1)
        self.lines = lines
        self.i = 0
        self._documents()

    def _documents(self) -> None:
        """Allow one ``---`` before the document and one ``...`` after it."""
        seen_start = False
        for n, raw in enumerate(self.lines):
            if raw.rstrip() == "---" or raw.startswith("--- "):
                if seen_start or any(_significant(r) for r in self.lines[:n]):
                    raise YAMLError("multi-document streams are outside the YAML "
                                    "Hermes reads", n + 1)
                seen_start = True
                self.lines[n] = raw[4:] if raw.startswith("--- ") else ""
            elif raw.rstrip() == "...":
                if any(_significant(r) for r in self.lines[n + 1:]):
                    raise YAMLError("multi-document streams are outside the YAML "
                                    "Hermes reads", n + 1)
                self.lines[n:] = []
                break

    # -- cursor over lines ----------------------------------------------------
    def next_significant(self) -> int | None:
        while self.i < len(self.lines) and not _significant(self.lines[self.i]):
            self.i += 1
        return self.i if self.i < len(self.lines) else None

    def document(self) -> Any:
        at = self.next_significant()
        if at is None:
            return None
        value = self.node(_indent(self.lines[at], at + 1))
        at = self.next_significant()
        if at is not None:
            raise YAMLError("unexpected content after the document; is it "
                            "indented less than what came before it?", at + 1)
        return value

    def node(self, indent: int) -> Any:
        at = self.next_significant()
        if at is None:
            return None
        raw = self.lines[at]
        column = _indent(raw, at + 1)
        content = raw[column:]
        if column < indent:
            return None
        if content == "-" or content.startswith(("- ", "-\t")):
            return self.sequence(column)
        if _split_key(_strip_comment(content), at + 1) is not None:
            return self.mapping(column)
        return self.inline(column - 1)

    def sequence(self, indent: int) -> list:
        items: list = []
        while True:
            at = self.next_significant()
            if at is None:
                return items
            raw = self.lines[at]
            column = _indent(raw, at + 1)
            content = raw[column:]
            if column != indent or not (content == "-" or content.startswith(("- ", "-\t"))):
                if column > indent:
                    raise YAMLError("a sequence item is indented past its siblings", at + 1)
                return items
            rest = content[1:]
            body = rest.lstrip(" \t")
            if not _strip_comment(body):
                self.i = at + 1
                items.append(self.node(indent + 1))
                continue
            # An item that opens a mapping or another sequence: the dash is
            # indentation for what follows it, so rewrite it as such and read
            # the block at the column its content starts in.
            inner = indent + 1 + (len(rest) - len(body))
            self.lines[at] = " " * inner + body
            stripped = _strip_comment(body)
            if (stripped == "-" or stripped.startswith(("- ", "-\t"))
                    or _split_key(stripped, at + 1) is not None):
                items.append(self.node(inner))
            else:
                items.append(self.inline(indent))

    def mapping(self, indent: int) -> dict:
        out: dict = {}
        while True:
            at = self.next_significant()
            if at is None:
                return out
            raw = self.lines[at]
            column = _indent(raw, at + 1)
            if column < indent:
                return out
            if column > indent:
                raise YAMLError("a key is indented past its siblings", at + 1)
            content = raw[column:]
            if content == "-" or content.startswith(("- ", "-\t")):
                return out      # a sequence that closes the enclosing block
            split = _split_key(content, at + 1)
            if split is None:
                raise YAMLError("expected a 'key: value' line, found "
                                f"{_strip_comment(content)[:40]!r}",
                                at + 1)
            key, offset = split
            _hashable(key, at + 1)
            rest = _strip_comment(content[offset:].strip())
            if not rest:
                self.i = at + 1
                nxt = self.next_significant()
                if nxt is None:
                    out[key] = None
                    continue
                child_raw = self.lines[nxt]
                child = _indent(child_raw, nxt + 1)
                child_content = child_raw[child:]
                if child > indent:
                    out[key] = self.node(child)
                elif child == indent and (child_content == "-"
                                          or child_content.startswith(("- ", "-\t"))):
                    out[key] = self.sequence(child)   # YAML's indentless sequence
                else:
                    out[key] = None
                continue
            # The value starts on this line: leave only the value on it, so
            # the inline reader sees what follows the key and nothing else.
            self.lines[at] = " " * (indent + 1) + raw[column + offset:].strip()
            out[key] = self.inline(indent)

    def inline(self, parent: int) -> Any:
        """A value that starts on the current line: scalar, flow or block scalar.

        *parent* is the indentation of the structure the value belongs to;
        continuation lines must be indented past it.
        """
        at = self.next_significant()
        raw = self.lines[at]
        text = raw.strip()
        stripped = _strip_comment(text)
        if not stripped:
            raise YAMLError("expected a value", at + 1)
        if stripped[0] in "&*!%@`":
            raise YAMLError(f"{stripped[:1]!r} is outside the YAML Hermes reads", at + 1)
        if stripped[:1] in "|>":
            return self.block_scalar(stripped, parent, at)
        if stripped[:1] in "[{":
            return self.flow(parent, at)
        if stripped[:1] in ("'", '"'):
            return self.quoted(parent, at)
        # A plain scalar, possibly continued on more deeply indented lines.
        parts = [stripped]
        self.i = at + 1
        while True:
            nxt = self.next_significant()
            if nxt is None:
                break
            more = self.lines[nxt]
            if _indent(more, nxt + 1) <= parent:
                break
            piece = _strip_comment(more.strip())
            if _split_key(piece, nxt + 1) is not None or piece.startswith("- "):
                raise YAMLError("a value continues onto a line that looks like a key; "
                                "is it indented too far?", nxt + 1)
            parts.append(piece)
            self.i = nxt + 1
        value = " ".join(parts)
        if ": " in value or value.endswith(":"):
            raise YAMLError(f"a plain value may not contain ': ' -- quote it: {value[:40]!r}",
                            at + 1)
        return resolve(value)

    def _gather(self, parent: int, at: int, done) -> tuple[str, int]:
        """Join lines from *at* until ``done(text)``; return text and next line."""
        text = self.lines[at].strip()
        end = at + 1
        while not done(text):
            if end >= len(self.lines):
                raise YAMLError("an unterminated flow collection or quoted string", at + 1)
            more = self.lines[end]
            if more.strip() and _indent(more, end + 1) <= parent and parent >= 0:
                raise YAMLError("a flow collection or quoted string continues at "
                                "too shallow an indent", end + 1)
            text += "\n" + more.strip()
            end += 1
        return text, end

    def flow(self, parent: int, at: int) -> Any:
        def balanced(text: str) -> bool:
            try:
                cursor = _Text(text, at + 1)
                _flow_value(cursor)
                return True
            except YAMLError as exc:
                return "unterminated" not in str(exc) and _raise(exc)

        text, end = self._gather(parent, at, balanced)
        cursor = _Text(text, at + 1)
        value = _flow_value(cursor)
        cursor.skip_space()
        if cursor.peek():
            raise YAMLError(f"unexpected {cursor.text[cursor.pos:][:20]!r} after a flow "
                            "collection", at + 1)
        self.i = end
        return value

    def quoted(self, parent: int, at: int) -> Any:
        def closed(text: str) -> bool:
            try:
                _quoted(_Text(text, at + 1))
                return True
            except YAMLError as exc:
                return "unterminated" not in str(exc) and _raise(exc)

        text, end = self._gather(parent, at, closed)
        cursor = _Text(text, at + 1)
        value = _quoted(cursor)
        cursor.skip_space()
        if cursor.peek():
            raise YAMLError(f"unexpected {cursor.text[cursor.pos:][:20]!r} after a quoted "
                            "string", at + 1)
        self.i = end
        return value

    def block_scalar(self, header: str, parent: int, at: int) -> str:
        match = re.fullmatch(r"([|>])([+-]?)([1-9]?)|([|>])([1-9])([+-])", header)
        if not match:
            raise YAMLError(f"a block scalar header {header!r} this reader does not know",
                            at + 1)
        style = match.group(1) or match.group(4)
        chomp = match.group(2) or match.group(6) or ""
        explicit = match.group(3) or match.group(5)
        body: list[str] = []
        end = at + 1
        width = parent + int(explicit) + 1 if explicit else None
        while end < len(self.lines):
            raw = self.lines[end]
            if raw.strip():
                column = _indent(raw, end + 1)
                if column <= parent:
                    break
                if width is None:
                    width = column
                if column < width:
                    break
                body.append(raw[width:])
            else:
                body.append("")
            end += 1
        self.i = end
        while body and body[-1] == "" and chomp != "+":
            body.pop()
        if style == "|":
            text = "\n".join(body)
        else:
            text, previous_blank = "", True
            for line in body:
                if not line:
                    text += "\n"
                    previous_blank = True
                elif line.startswith((" ", "\t")):
                    text += ("" if previous_blank else "\n") + line + "\n"
                    previous_blank = True
                else:
                    text += ("" if previous_blank else " ") + line
                    previous_blank = False
            text = text.rstrip("\n") if not text.endswith("\n\n") else text
        if not body:
            return ""
        return text + ("\n" if chomp != "-" else "")


def _raise(exc: YAMLError) -> bool:
    raise exc


def load(text: str) -> Any:
    """Parse one YAML document of the subset Hermes uses.

    Returns ``None`` for an empty document, as ``yaml.safe_load`` does.
    """
    return _Reader(str(text)).document()


# -- writing -------------------------------------------------------------------

_INDICATORS = set("-?:,[]{}#&*!|>'\"%@`")

#: Inside ``[...]`` or ``{...}`` a plain scalar is written only when it is
#: made of characters no flow parser can mistake for structure. Quoting more
#: than strictly necessary costs a pair of quotes; quoting less costs a file.
_FLOW_PLAIN = re.compile(r"^[A-Za-z0-9_./(][A-Za-z0-9_./+\-~=() ]*(?<! )$")


def _plain_ok(text: str, in_flow: bool) -> bool:
    if not text or text != text.strip() or resolve(text) != text:
        return False
    if text in ("=", "<<"):
        return False        # YAML 1.1's value and merge keys, not strings
    if text[0] in _INDICATORS and not (text[0] == "-" and len(text) > 1
                                       and text[1] not in " \t"):
        return False
    if ": " in text or " #" in text or text.endswith(":") or "\t" in text:
        return False
    if any(_unprintable(c) for c in text) or text.startswith(("---", "...")):
        return False
    if in_flow and not _FLOW_PLAIN.match(text):
        return False
    return True


def _unprintable(char: str) -> bool:
    code = ord(char)
    return (code < 32 or code == 0x7f or char in "\x85\u2028\u2029\ufeff"
            or 0xD800 <= code <= 0xDFFF)


_WRITE_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\t": "\\t", "\r": "\\r"}


def _double(value: str) -> str:
    """*value* double-quoted, every character that could break a line escaped."""
    out = []
    for char in value:
        if char in _WRITE_ESCAPES:
            out.append(_WRITE_ESCAPES[char])
        elif _unprintable(char):
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def _float(value: float) -> str:
    if math.isnan(value):
        return ".nan"
    if math.isinf(value):
        return ".inf" if value > 0 else "-.inf"
    text = repr(value)
    if "e" in text and "." not in text.split("e")[0]:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}.0e{exponent}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        if exponent[0] not in "+-":
            exponent = "+" + exponent
        text = f"{mantissa}e{exponent}"
    return text


def _scalar(value: Any, in_flow: bool = False) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, str):
        if _plain_ok(value, in_flow):
            return value
        if "'" not in value and not any(_unprintable(c) for c in value):
            return "'" + value + "'"
        return _double(value)
    raise YAMLError(f"cannot write a {type(value).__name__} as YAML")


def flow(value: Any, _nested: bool = False) -> str:
    """*value* as one line of YAML that reads back to it.

    A scalar is written for the block context it lands in after ``key: ``, so
    a URL stays unquoted; anything inside a collection is written for the
    stricter flow context.
    """
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_scalar(k, True)}: {flow(v, True)}"
                               for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(flow(v, True) for v in value) + "]"
    return _scalar(value, in_flow=_nested)


def _block(value: Any, indent: int, sort_keys: bool) -> list[str]:
    pad = " " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0])) if sort_keys \
            else value.items()
        for key, item in items:
            name = _scalar(key)
            if isinstance(item, (dict, list, tuple)) and item:
                lines.append(f"{pad}{name}:")
                lines += _block(item, indent + 2, sort_keys)
            else:
                lines.append(f"{pad}{name}: {flow(item) if isinstance(item, (dict, list, tuple)) else _scalar(item)}")
        return lines
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (dict, list, tuple)) and item:
                inner = _block(item, indent + 2, sort_keys)
                lines.append(f"{pad}- {inner[0][indent + 2:]}")
                lines += inner[1:]
            else:
                lines.append(f"{pad}- {flow(item) if isinstance(item, (dict, list, tuple)) else _scalar(item)}")
        return lines
    return [pad + _scalar(value)]


def dump(value: Any, sort_keys: bool = False) -> str:
    """*value* as block-style YAML, the way a person would write the config."""
    if isinstance(value, (dict, list, tuple)) and not value:
        return flow(value) + "\n"
    return "\n".join(_block(value, 0, sort_keys)) + "\n"
