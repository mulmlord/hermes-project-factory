"""Held-out acceptance tests for the stencil task. Never shown to the factory."""
from types import SimpleNamespace

import testkit

import stencil
from stencil import Template, TemplateSyntaxError, UndefinedError, render


def test_plain_text_is_copied_verbatim():
    source = "  line one\n\tline two  \n"
    assert render(source) == source


def test_output_and_keyword_arguments_win():
    assert render("Hi {{ name }}!", {"name": "Ada"}) == "Hi Ada!"
    assert render("{{ a }}{{ b }}", {"a": 1, "b": 2}, b=3) == "13"


def test_none_renders_empty():
    assert render("[{{ x }}]", x=None) == "[]"


def test_dotted_lookup_prefers_keys_then_attributes():
    obj = SimpleNamespace(name="attr")
    assert render("{{ a.name }}", a={"name": "key"}) == "key"
    assert render("{{ a.name }}", a=obj) == "attr"
    assert render("{{ a.b.c }}", a={"b": SimpleNamespace(c=7)}) == "7"


def test_literals():
    assert render("{{ \"double\" }} {{ 'single' }} {{ 42 }} {{ -3 }}") == "double single 42 -3"


def test_if_elif_else():
    source = "{% if n > 10 %}big{% elif n > 5 %}medium{% else %}small{% endif %}"
    assert [render(source, n=n) for n in (11, 6, 1)] == ["big", "medium", "small"]


def test_boolean_precedence_and_parentheses():
    assert render("{% if not a or b and c %}y{% else %}n{% endif %}",
                  a=True, b=True, c=False) == "n"
    assert render("{% if not (a and b) %}y{% else %}n{% endif %}",
                  a=True, b=False) == "y"
    assert render("{% if x == 'go' and y != 2 %}y{% endif %}", x="go", y=3) == "y"


def test_comparisons():
    source = "{% if a <= b %}le{% endif %}{% if a >= b %}ge{% endif %}{% if a < b %}lt{% endif %}"
    assert render(source, a=2, b=2) == "lege"
    assert render(source, a=1, b=2) == "lelt"


def test_for_loop_with_loop_variable():
    source = "{% for x in xs %}{{ loop.index }}:{{ x }}{% if not loop.last %},{% endif %}{% endfor %}"
    assert render(source, xs=["a", "b", "c"]) == "1:a,2:b,3:c"


def test_loop_first():
    source = "{% for x in xs %}{% if loop.first %}[{% endif %}{{ x }}{% endfor %}"
    assert render(source, xs=[1, 2]) == "[12"


def test_for_else_on_empty():
    source = "{% for x in xs %}{{ x }}{% else %}none{% endfor %}"
    assert render(source, xs=[]) == "none"
    assert render(source, xs=[5]) == "5"


def test_nested_loops_have_their_own_loop():
    source = ("{% for row in rows %}{% for c in row %}{{ loop.index }}{% endfor %}"
              "/{{ loop.index }};{% endfor %}")
    assert render(source, rows=[[1, 2], [3]]) == "12/1;1/2;"


def test_loop_variable_does_not_leak():
    source = "{% for x in xs %}{{ x }}{% endfor %}{{ x }}"
    assert render(source, xs=[1, 2], x="outer") == "12outer"
    with testkit.raises(UndefinedError):
        render("{% for x in xs %}{% endfor %}{{ x }}", xs=[1])


def test_comments_produce_nothing():
    assert render("a{# one\ntwo #}b") == "ab"


def test_filters_and_chaining():
    assert render("{{ s | trim | upper }}", s="  hi ") == "HI"
    assert render("{{ s | title }}", s="the end") == "The End"
    assert render("{{ s | lower }}", s="ABC") == "abc"
    assert render("{{ xs | length }}", xs=[1, 2, 3]) == "3"
    assert render("{{ xs | join(', ') }}", xs=[1, 2]) == "1, 2"


def test_filter_binds_tighter_than_comparison():
    assert render("{% if xs | length == 2 %}two{% endif %}", xs="ab") == "two"


def test_default():
    assert render("{{ missing | default('x') }}") == "x"
    assert render("{{ v | default('x') }}", v=None) == "x"
    assert render("{{ v | default('x') }}", v=0) == "0"


def test_undefined_names_raise_when_used():
    for source in ("{{ nope }}", "{{ nope | upper }}", "{% for x in nope %}{% endfor %}",
                   "{{ d.missing }}"):
        with testkit.raises(UndefinedError):
            render(source, d={})


def test_undefined_is_false_in_conditions():
    assert render("{% if nope %}y{% elif also.nope %}z{% else %}n{% endif %}") == "n"


def test_escape_filter():
    assert render("{{ s | escape }}", s="<a href=\"x\">&'</a>") == (
        "&lt;a href=&quot;x&quot;&gt;&amp;&#x27;&lt;/a&gt;")


def test_autoescape():
    page = Template("{{ a }}|{{ b | safe }}|{{ c | escape }}", autoescape=True)
    assert page.render(a="<i>", b="<b>", c="<u>") == "&lt;i&gt;|<b>|&lt;u&gt;"
    assert Template("{{ a }}").render(a="<i>") == "<i>"


def test_syntax_errors_are_raised_at_construction_with_a_line():
    cases = {
        "ok\n{{ name": 2,
        "{% if a %}\n\n{% bogus %}{% endif %}": 3,
        "x\n{% endif %}": 2,
        "{% for a in b %}\n{% else %}\n{% else %}{% endfor %}": 3,
        "\n\n{% if a %}never closed": 3,
        "{{ a | nosuchfilter }}": 1,
        "a\nb\n{{ 1 + }}": 3,
        "{# open": 1,
    }
    for source, line in cases.items():
        with testkit.raises(TemplateSyntaxError) as caught:
            Template(source)
        assert caught.value.line == line, source


def test_exports():
    assert stencil.Template is Template and stencil.render is render
    assert issubclass(TemplateSyntaxError, Exception)
    assert issubclass(UndefinedError, Exception)
