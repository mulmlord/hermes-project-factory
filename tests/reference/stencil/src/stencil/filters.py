import html


class Markup(str):
    """A string that autoescaping must leave alone."""


def escape(value):
    if isinstance(value, Markup):
        return value
    return Markup(html.escape(str(value), quote=True))


FILTERS = {
    "upper": lambda v: str(v).upper(),
    "lower": lambda v: str(v).lower(),
    "title": lambda v: str(v).title(),
    "trim": lambda v: str(v).strip(),
    "length": len,
    "join": lambda v, sep: sep.join(str(i) for i in v),
    "default": lambda v, d: d if v is None else v,
    "escape": escape,
    "safe": lambda v: Markup("" if v is None else str(v)),
}
