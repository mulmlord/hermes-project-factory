from .errors import TemplateSyntaxError

CLOSERS = {"{{": "}}", "{%": "%}", "{#": "#}"}


def tokenize(source):
    """Yield (kind, text, line) with kind one of text/output/tag/comment."""
    tokens, at, line = [], 0, 1
    while at < len(source):
        starts = [i for i in (source.find(o, at) for o in CLOSERS) if i >= 0]
        start = min(starts) if starts else len(source)
        if start > at:
            tokens.append(("text", source[at:start], line))
            line += source.count("\n", at, start)
        if start == len(source):
            break
        opener = source[start:start + 2]
        end = source.find(CLOSERS[opener], start + 2)
        if end < 0:
            raise TemplateSyntaxError(f"unclosed {opener}", line)
        kind = {"{{": "output", "{%": "tag", "{#": "comment"}[opener]
        tokens.append((kind, source[start + 2:end].strip(), line))
        line += source.count("\n", start, end + 2)
        at = end + 2
    return tokens
