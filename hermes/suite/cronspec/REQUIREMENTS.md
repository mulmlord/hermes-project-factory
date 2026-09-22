# cronspec — requirements

Build `cronspec`, an installable Python library (Python 3.10+, standard library
only) that parses five-field cron expressions and computes when they fire.

## Layout

The import package is `cronspec`, placed at `src/cronspec/`, with these modules:

- `cronspec/__init__.py` — re-exports `parse`, `Cron` and `CronError`
- `cronspec/fields.py` — the five field definitions and their value ranges
- `cronspec/parser.py` — turning an expression into a `Cron`
- `cronspec/schedule.py` — matching and next-occurrence search
- `cronspec/cli.py` and `cronspec/__main__.py` — the command line interface

Ship a `pyproject.toml` so `pip install .` installs the package, and a test
suite of your own under `tests/`.

## Expressions

`cronspec.parse(expression: str) -> Cron` accepts five fields separated by
runs of whitespace, in this order:

| field        | range | names                         |
|--------------|-------|-------------------------------|
| minute       | 0–59  |                               |
| hour         | 0–23  |                               |
| day of month | 1–31  |                               |
| month        | 1–12  | `JAN` … `DEC`                 |
| day of week  | 0–7   | `SUN` … `SAT`; 0 and 7 are Sunday |

Each field is a comma-separated list of items. An item is one of:

- `*` — every value in the range
- `N` — a single value, or a name where the field has names (case-insensitive)
- `A-B` — an inclusive range, `A <= B`; names are allowed at either end
- `*/S` or `A-B/S` — every `S`-th value of the range, starting at its low end
- `A/S` — every `S`-th value from `A` to the top of the range

Leading and trailing whitespace around the whole expression is ignored.

These macros replace the whole expression:

| macro                    | equivalent  |
|--------------------------|-------------|
| `@yearly`, `@annually`   | `0 0 1 1 *` |
| `@monthly`               | `0 0 1 * *` |
| `@weekly`                | `0 0 * * 0` |
| `@daily`, `@midnight`    | `0 0 * * *` |
| `@hourly`                | `0 * * * *` |

## The Cron object

`Cron` has these attributes, each a `frozenset` of `int`:

- `minutes`, `hours`, `days` (day of month), `months`
- `weekdays` — 0 to 6, where 0 is Sunday; a 7 in the expression is stored as 0

and these methods:

- `matches(when: datetime) -> bool` — whether the schedule fires in the minute
  containing `when` (seconds and microseconds are ignored)
- `next(after: datetime) -> datetime` — the first firing time strictly after
  `after`, with seconds and microseconds set to zero
- `upcoming(after: datetime, count: int) -> list[datetime]` — the next `count`
  firing times, in order, each strictly after the previous one

`str(cron)` returns the expression that was parsed, with macros kept as
written and surrounding whitespace removed.

Datetimes are naive. Treat them as wall-clock time and never convert zones.

### Day matching

A field written as exactly `*` is *unrestricted*; any other spelling,
including `*/2`, restricts it. When day of month and day of week are both
restricted, a day matches if **either** matches. Otherwise a day matches only
if **both** match (an unrestricted field matches every day).

So `0 0 13 * FRI` fires on every 13th *and* every Friday, while
`0 0 13 * *` fires only on the 13th.

## Errors

Every malformed expression raises `cronspec.CronError`, a subclass of
`ValueError`, and its message names the field at fault using the words in the
table above (`minute`, `hour`, `day of month`, `month`, `day of week`).
Malformed means: the wrong number of fields, an empty item, a value or name
out of range or unknown, a step of zero, a range whose start is above its end,
or an unknown macro (whose message need not name a field).

`next` and `upcoming` raise `CronError` when no firing time exists within
five years of `after` — for example `0 0 30 2 *`, which asks for 30 February.

## Command line

`python -m cronspec EXPRESSION [--after YYYY-MM-DDTHH:MM] [--count N]`

prints the next `N` (default 5) firing times after `--after` (default: now),
one per line, formatted `YYYY-MM-DDTHH:MM`, and exits 0. A malformed
expression prints the error message to standard error and exits 2.
