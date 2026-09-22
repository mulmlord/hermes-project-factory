from .fields import FIELDS, MACROS


class CronError(ValueError):
    pass


def _value(text, spec):
    upper = text.upper()
    if upper in spec.names:
        return spec.names[upper]
    if not text.isdigit():
        raise CronError(f"{spec.name}: unknown value {text!r}")
    number = int(text)
    if not spec.low <= number <= spec.high:
        raise CronError(f"{spec.name}: {number} outside {spec.low}-{spec.high}")
    return number


def _item(text, spec):
    if not text:
        raise CronError(f"{spec.name}: empty item")
    base, _, step_text = text.partition("/")
    step = 1
    if _:
        if not step_text.isdigit() or int(step_text) == 0:
            raise CronError(f"{spec.name}: bad step {step_text!r}")
        step = int(step_text)
    if base == "*":
        low, high = spec.low, spec.high
    elif "-" in base:
        a, _, b = base.partition("-")
        low, high = _value(a, spec), _value(b, spec)
        if low > high:
            raise CronError(f"{spec.name}: range {base!r} runs backwards")
    else:
        low = _value(base, spec)
        high = spec.high if _ else low
    return set(range(low, high + 1, step))


def parse(expression):
    from .schedule import Cron

    text = expression.strip()
    source = text
    if text.startswith("@"):
        if text.lower() not in MACROS:
            raise CronError(f"unknown macro {text!r}")
        text = MACROS[text.lower()]
    parts = text.split()
    if len(parts) != 5:
        raise CronError(f"expected 5 fields, got {len(parts)}")
    sets = []
    for part, spec in zip(parts, FIELDS):
        values = set()
        for item in part.split(","):
            values |= _item(item, spec)
        sets.append(values)
    weekdays = {0 if d == 7 else d for d in sets[4]}
    return Cron(source, *map(frozenset, sets[:4]), frozenset(weekdays),
                parts[2] != "*", parts[4] != "*")
