from dataclasses import dataclass, field


@dataclass(frozen=True)
class Field:
    name: str
    low: int
    high: int
    names: dict = field(default_factory=dict)


_MONTHS = ("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC").split()
_DAYS = ("SUN MON TUE WED THU FRI SAT").split()

FIELDS = (
    Field("minute", 0, 59),
    Field("hour", 0, 23),
    Field("day of month", 1, 31),
    Field("month", 1, 12, {n: i + 1 for i, n in enumerate(_MONTHS)}),
    Field("day of week", 0, 7, {n: i for i, n in enumerate(_DAYS)}),
)

MACROS = {
    "@yearly": "0 0 1 1 *", "@annually": "0 0 1 1 *", "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0", "@daily": "0 0 * * *", "@midnight": "0 0 * * *",
    "@hourly": "0 * * * *",
}
