# stencil — requirements

Build `stencil`, an installable Python library (Python 3.10+, standard library
only) implementing a small text template language.

## Layout

The import package is `stencil`, placed at `src/stencil/`, with these modules:

- `stencil/__init__.py` — re-exports `Template`, `render`,
  `TemplateSyntaxError` and `UndefinedError`
- `stencil/errors.py` — the two exception types
- `stencil/lexer.py` — splitting source text into text, output, tag and
  comment tokens, each carrying the line it starts on
- `stencil/parser.py` — building a tree of nodes from the tokens
- `stencil/filters.py` — the built-in filters
- `stencil/render.py` — evaluating the tree against a context

Ship a `pyproject.toml` so `pip install .` installs the package, and a test
suite of your own under `tests/`.

## API

- `Template(source: str, autoescape: bool = False)` parses the source
  immediately; every syntax error is raised here, never during rendering.
- `Template.render(context: dict | None = None, **names) -> str` renders the
  template. Keyword arguments are added to the context and win over it.
- `render(source: str, context: dict | None = None, **names) -> str` is
  shorthand for `Template(source).render(context, **names)`.
- `TemplateSyntaxError(Exception)` has an attribute `line`, the 1-based line
  number of the tag or expression at fault.
- `UndefinedError(Exception)` is raised when rendering needs a name the
  context does not define.

## Syntax

Everything outside the delimiters below is copied to the output unchanged,
including all whitespace and newlines.

- `{{ expression }}` — output the value of the expression
- `{% if expr %} … {% elif expr %} … {% else %} … {% endif %}` — `elif` and
  `else` are optional and `elif` may repeat
- `{% for name in expr %} … {% else %} … {% endfor %}` — the `else` body is
  rendered when the iterable is empty
- `{# comment #}` — produces no output; may span lines

Tags nest to any depth.

### Expressions

- names and dotted paths: `user.name` looks up `name` on `user`, trying a
  mapping key first and an attribute second
- literals: integers (`42`, `-3`) and strings in single or double quotes
- comparisons: `==`, `!=`, `<`, `>`, `<=`, `>=`
- boolean operators: `not`, `and`, `or`, with the usual precedence
  (`not` binds tightest, then `and`, then `or`), and parentheses
- filters: `expr | name` or `expr | name(arg, ...)`, chained left to right;
  a filter binds tighter than any operator

### Output

`None` renders as the empty string; every other value renders as `str(value)`.

### Loops

Inside a `for` body the name `loop` is defined with:

- `loop.index` — 1-based position
- `loop.first`, `loop.last` — booleans

The loop variable and `loop` are visible only inside the body; after the loop
the names they shadowed are visible again. Nested loops each have their own
`loop`.

### Undefined names

Output of an undefined name, a filter applied to one (other than `default`),
iterating over one, or a dotted lookup on one raises `UndefinedError`. A
dotted lookup that finds neither a key nor an attribute is undefined too.

In an `if` or `elif` condition an undefined name is simply false.

## Filters

| filter            | effect                                                   |
|-------------------|----------------------------------------------------------|
| `upper`, `lower`, `title` | the `str` methods of the same name              |
| `trim`            | strip leading and trailing whitespace                    |
| `length`          | `len(value)`                                             |
| `join(sep)`       | `sep.join(str(item) for item in value)`                  |
| `default(value)`  | the argument when the input is undefined or `None`       |
| `escape`          | HTML-escape `& < > " '` as `&amp; &lt; &gt; &quot; &#x27;` |
| `safe`            | mark the value as not to be escaped (see below)          |

A filter name not in this table is a `TemplateSyntaxError`.

## Autoescaping

With `autoescape=True`, every `{{ }}` output is HTML-escaped as `escape` does,
unless the value passed through `safe`. A value escaped by `escape` is not
escaped a second time. With `autoescape=False` (the default) nothing is
escaped except by an explicit `escape`.

## Syntax errors

Raise `TemplateSyntaxError`, with `line` set to the line where the offending
delimiter starts, for: an unclosed `{{`, `{%` or `{#`; an unknown tag; an
`elif`, `else`, `endif` or `endfor` with no matching opener; a second `else`
in one block, or an `elif` after the `else`; a block left
open at the end of the template (report the line of its opening tag); a
malformed expression; and an unknown filter.
