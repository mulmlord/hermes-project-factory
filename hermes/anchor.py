"""Locating the text a search/replace edit meant to change.

A model that sends an edit is copying an anchor out of a file it was shown
earlier in its context. It gets the anchor slightly wrong far more often than
it gets the *change* wrong: a tab became spaces, a trailing space was dropped,
the snippet was re-indented on its way through JSON. Refusing those costs a
whole generation to re-ask a question the model already answered correctly.

So the anchor is looked for in widening circles, and the first circle that
holds exactly one match wins:

    exact text                      -> apply there
    exact text, several matches     -> only with a hint saying which
    whole lines, ignoring indent    -> apply there, re-indented to the file
    whole lines, several matches    -> only with a hint saying which
    nothing                         -> refuse, and say so

What never widens is uniqueness. Two candidate sites and no hint is refused
every time, because guessing which of two identical blocks was meant is how
working code is silently destroyed -- and a refusal is recoverable while a
wrong edit measured as a pass is not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: Words that introduce a named thing in the languages a project might be in.
#: Used only to turn a symbol hint into a line number, never to parse code.
_DEFINES = re.compile(
    r"\b(def|class|fn|func|function|struct|enum|impl|interface|trait|type|"
    r"module|namespace|package|const|var|let|template|record)\b")


@dataclass(frozen=True)
class Anchor:
    """Where an edit lands, and the replacement as the file wants it written.

    *start* and *end* are character offsets into the file, so the caller
    splices rather than searching again. *replacement* is the edit's own
    replacement text after any re-indentation the match implied. *how* names
    the circle that matched, for the log.
    """

    start: int
    end: int
    replacement: str
    how: str = "exact"

    @property
    def exact(self) -> bool:
        """True when the model's anchor matched the file character for character."""
        return self.how == "exact"

    def splice(self, text: str) -> str:
        return text[:self.start] + self.replacement + text[self.end:]


def locate(text: str, find: str, replace: str, *, line: int | None = None,
           symbol: str = "") -> tuple[Anchor | None, str]:
    """Find where *find* sits in *text*, or say why it cannot be placed.

    *line* and *symbol* are optional hints the model may send about which
    occurrence it meant; they only ever break a tie between matches that are
    otherwise equally good, and never create a match of their own.

    Returns the anchor and an empty reason, or None and a sentence addressed to
    whoever sent the edit.
    """
    if not find:
        return None, "the \"find\" text is empty, so there is nothing to look for."
    hint = _hint_line(text, line, symbol)

    hits = _exact_hits(text, find)
    if hits:
        chosen = hits[0] if len(hits) == 1 else _nearest(text, hits, hint)
        if chosen is None:
            return None, (
                f"the text in \"find\" appears {len(hits)} times, so the edit is "
                "ambiguous. Include more surrounding lines so it occurs once, or "
                "name the function or class it is inside in \"symbol\".")
        return Anchor(chosen, chosen + len(find), replace,
                      "exact" if len(hits) == 1 else "hint"), ""

    spans = _line_spans(text, find)
    if not spans:
        if _line_spans(text, find, shaped=False):
            return None, (
                "those lines are in the file, but they are not nested the way "
                "\"find\" nests them, so which code you meant is a guess. Copy the "
                "lines exactly, keeping the leading whitespace.")
        return None, (
            "the text in \"find\" does not appear in the file. Copy it exactly "
            "from the current contents shown to you, or return the whole file "
            "under \"files\".")
    span = spans[0] if len(spans) == 1 else _nearest(text, spans, hint, key=lambda s: s[0])
    if span is None:
        return None, (
            f"the text in \"find\" matches {len(spans)} places once indentation is "
            "ignored, so the edit is ambiguous. Include more surrounding lines so "
            "it occurs once, or name the function or class it is inside in "
            "\"symbol\".")
    start, end = span
    return Anchor(start, end, _reindent(text[start:end], replace), "whitespace"), ""


def _exact_hits(text: str, find: str) -> list[int]:
    """Every offset at which *find* occurs verbatim."""
    hits, at = [], text.find(find)
    while at >= 0:
        hits.append(at)
        at = text.find(find, at + 1)
    return hits


def _line_spans(text: str, find: str, shaped: bool = True) -> list[tuple[int, int]]:
    """Spans of whole lines equal to *find* once indentation and trailing space go.

    The span covers the matched lines as the file has them, without the newline
    that ends the last one, so a replacement that does not end in a newline
    cannot glue two lines together.

    *shaped* also requires the block to nest the way *find* nests: the same
    lines at the same depths, whether the file spells a depth with a tab or
    four spaces. Only the spelling of indentation is forgiven, never the
    structure, because a body that sits one level in is different code from the
    same body at the top of a function and nothing in the edit says which was
    meant. Clearing the flag answers the separate question of whether the lines
    are there at all, which is what makes the refusal specific.
    """
    # A blank line matches like any other line rather than being trimmed away:
    # the span has to cover exactly the lines "find" described, or the
    # replacement that stands for them lands one line out.
    kept = find.splitlines()
    needle = [part.strip() for part in kept]
    if not any(needle):
        return []
    depths = _depths(kept)

    lines = text.splitlines(keepends=True)
    bare = [part.strip() for part in lines]
    offsets, at = [], 0
    for part in lines:
        offsets.append(at)
        at += len(part)

    spans = []
    for first in range(len(lines) - len(needle) + 1):
        window = lines[first:first + len(needle)]
        if bare[first:first + len(needle)] != needle:
            continue
        if shaped and _depths([p.rstrip("\r\n") for p in window]) != depths:
            continue
        last = first + len(needle) - 1
        spans.append((offsets[first], offsets[last] + len(lines[last].rstrip("\r\n"))))
    return spans


def _depths(lines: list[str]) -> tuple[int, ...]:
    """How far each line is nested past the block, counted in the block's unit.

    Measuring in the block's own unit is what lets a tab-indented file match a
    space-indented anchor: both say "one level in", and only the spelling of
    that level differs.
    """
    block = "\n".join(lines)
    base, unit = len(_common_indent(block)), _unit(block)
    if not unit:
        return tuple(0 for _ in lines)
    return tuple(0 if not part.strip()
                 else (len(part) - len(part.lstrip()) - base) // len(unit)
                 for part in lines)


def _reindent(matched: str, replace: str) -> str:
    """*replace* moved to sit at the indentation the matched block sits at.

    The replacement keeps its own internal shape -- only the block as a whole
    moves -- so a nested body stays nested relative to its own first line, and
    it is rejoined with the line ending the block it replaces already used.
    """
    into, base = _common_indent(matched), _common_indent(replace)
    unit, step = _unit(matched) or _unit(replace), _unit(replace)
    out = []
    for part in replace.splitlines():
        if not part.strip():
            out.append("")
            continue
        body = part[len(base):] if part.startswith(base) else part.lstrip()
        inner = body[:len(body) - len(body.lstrip())]
        if unit and step and unit != step:
            # The file indents with what the replacement does not -- tabs
            # against spaces. Copying the replacement's own whitespace would
            # leave the block indented two ways at once, so its nesting is
            # counted in levels and re-spelled in the file's unit.
            inner = unit * (len(inner) // len(step))
        out.append(into + inner + body.lstrip())
    return ("\r\n" if "\r\n" in matched else "\n").join(out)


def _unit(block: str) -> str:
    """One level of indentation as *block* spells it, judged by its own lines.

    That is the smallest step any line takes past the block's shared indent:
    a body nested once under its first line. An unindented block has no unit
    and nothing is inferred for it.
    """
    base = len(_common_indent(block))
    steps = [part[base:len(part) - len(part.lstrip())]
             for part in block.splitlines() if part.strip()]
    return min((s for s in steps if s), key=len, default="")


def _common_indent(block: str) -> str:
    """The leading whitespace every non-blank line of *block* shares."""
    shared: str | None = None
    for part in block.splitlines():
        if not part.strip():
            continue
        indent = part[:len(part) - len(part.lstrip())]
        if shared is None:
            shared = indent
            continue
        keep = 0
        while keep < min(len(shared), len(indent)) and shared[keep] == indent[keep]:
            keep += 1
        shared = shared[:keep]
    return shared or ""


def _hint_line(text: str, line: int | None, symbol: str) -> int | None:
    """The zero-based line the hints point at, or None when they point nowhere.

    A symbol that is defined in one place resolves to that definition; failing
    that, a symbol mentioned exactly once resolves to the mention. A symbol that
    appears all over the file is not a hint and is discarded rather than
    guessed from.
    """
    if isinstance(line, int) and line > 0:
        return min(line - 1, max(0, text.count("\n")))
    symbol = symbol.strip()
    if not symbol:
        return None
    lines = text.splitlines()
    word = re.compile(rf"\b{re.escape(symbol)}\b")
    mentions = [n for n, part in enumerate(lines) if word.search(part)]
    defined = [n for n in mentions if _DEFINES.search(lines[n])]
    if len(defined) == 1:
        return defined[0]
    return mentions[0] if len(mentions) == 1 else None


def _nearest(text: str, candidates: list, hint: int | None, key=lambda c: c):
    """The candidate whose line is closest to *hint*, or None if nothing decides.

    A tie is not a decision: two matches the same distance either side of the
    hint leave the edit exactly as ambiguous as it was without it.
    """
    if hint is None:
        return None
    scored = [(abs(text.count("\n", 0, key(c)) - hint), c) for c in candidates]
    scored.sort(key=lambda pair: pair[0])
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]
