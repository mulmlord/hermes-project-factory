"""The JSON shapes the factory asks for, as values a server can enforce.

Every agent reply is JSON, and until now the only thing holding it to a shape
was the example printed in the prompt: the model was *asked* for a key called
``tickets`` and Python found out afterwards whether it got one. A schema turns
that into a shape the decoder cannot leave -- vLLM and llama.cpp compile it to
a grammar and mask the tokens that would break it, ollama takes the same
document in ``format`` -- so a missing key stops being a lost generation.

The builders here exist to keep a schema short enough to sit beside the prompt
it describes in :mod:`hermes.prompts`. A schema far from its prompt drifts from
it, and a drifted schema is worse than none: it constrains the model to a shape
the caller no longer reads.

:func:`check` is the same document applied by hand, for the servers that will
not apply it themselves. It is deliberately forgiving about everything except
the keys the caller is going to read -- see :func:`obj`.
"""
from __future__ import annotations

from typing import Any, Sequence

#: How hard the server is asked to hold the reply to a shape. Weakest first:
#: a refusal walks down this list rather than falling off the end of it, so a
#: server that knows ``json_object`` but not ``json_schema`` keeps the half of
#: the guarantee it does speak.
OFF, JSON, SCHEMA = "off", "json", "schema"
LEVELS = (OFF, JSON, SCHEMA)

STRING: dict[str, Any] = {"type": "string"}
NUMBER: dict[str, Any] = {"type": "number"}
INTEGER: dict[str, Any] = {"type": "integer"}
BOOLEAN: dict[str, Any] = {"type": "boolean"}


def obj(properties: dict[str, Any], required: Sequence[str] = ()) -> dict[str, Any]:
    """An object with *properties*, of which *required* must be present.

    Required means "the caller cannot do its job without this", not "the
    prompt mentions it". Constrained decoding makes a required key mandatory
    for the decoder, so a long required list forces a model to fill in fields
    it has nothing to say about, and an invented ticket is worse than an
    absent one. Most prompts here require one or two keys and offer the rest.

    Additional properties are left allowed on purpose: a model that adds a
    ``notes`` key has not failed at anything, and forbidding it only turns a
    harmless extra into a refusal on the servers that enforce the schema.
    """
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = list(required)
    return schema


def arr(items: dict[str, Any] = STRING) -> dict[str, Any]:
    return {"type": "array", "items": items}


def enum(*values: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def any_of(*schemas: dict[str, Any]) -> dict[str, Any]:
    """Any one of *schemas*: the reply may take whichever shape it needs.

    There is one caller and it is the reason this exists: an agent holding
    lookup tools may answer the task *or* ask to read a file first, and a
    decoder constrained to the answer alone would take the choice away --
    forcing a verdict out of an agent that said it wanted to check something
    is precisely the guess :class:`hermes.tools.Lookup` was built to avoid.
    """
    return {"anyOf": [s for s in schemas if s]}


#: Free-form value: used where the model genuinely chooses the keys, such as
#: an experiment's parameter block.
ANY: dict[str, Any] = {}


_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "number": (int, float),
    "integer": int,
}


def check(schema: dict[str, Any] | None, value: Any, *, limit: int = 3) -> list[str]:
    """Where *value* departs from *schema*, in words a model can act on.

    This is the fallback path, not the main one: on a server that compiles the
    schema into a grammar the answer cannot be wrong and this returns nothing
    for its trouble. It earns its keep on the servers that ignore the
    parameter, where it converts "the caller crashed three phases later on a
    missing key" into a retry that says which key.

    Stops at *limit* complaints. The point is to give the retry one concrete
    thing to fix, not to produce a validation report nobody reads.
    """
    found: list[str] = []
    _walk(schema or {}, value, "reply", found, limit)
    return found


def _walk(schema: dict[str, Any], value: Any, path: str, found: list[str],
          limit: int) -> None:
    if len(found) >= limit or not schema:
        return
    if "anyOf" in schema:
        # Satisfying any branch is satisfying the schema. When none is
        # satisfied the closest one is reported: telling a model it failed
        # every alternative at once is not something it can act on.
        attempts = []
        for branch in schema["anyOf"]:
            complaints: list[str] = []
            _walk(branch, value, path, complaints, limit)
            if not complaints:
                return
            attempts.append(complaints)
        found.extend(min(attempts, key=len)[: limit - len(found)])
        return
    expected = schema.get("type")
    wanted = _TYPES.get(expected) if expected else None
    if wanted and not _is(value, expected, wanted):
        found.append(f"{path} should be a {expected}, not {_name(value)}")
        return
    if expected == "object":
        for key in schema.get("required", ()):
            if value.get(key) is None:
                found.append(f"{path} is missing the required key {key!r}")
                if len(found) >= limit:
                    return
        for key, sub in schema.get("properties", {}).items():
            if key in value and value[key] is not None:
                _walk(sub, value[key], f"{path}.{key}", found, limit)
    elif expected == "array":
        for index, item in enumerate(value):
            _walk(schema.get("items", {}), item, f"{path}[{index}]", found, limit)
            if len(found) >= limit:
                return
    choices = schema.get("enum")
    if choices and value not in choices:
        found.append(f"{path} should be one of {', '.join(choices)}, not {value!r}")


def _is(value: Any, expected: str, wanted: type | tuple[type, ...]) -> bool:
    # bool is an int in Python and nowhere else; a model that answers `true`
    # where a count belongs has made a real mistake and must be told.
    if isinstance(value, bool) and expected in ("number", "integer"):
        return False
    return isinstance(value, wanted)


def _name(value: Any) -> str:
    return {dict: "object", list: "array", str: "string", bool: "boolean",
            int: "number", float: "number", type(None): "null"}.get(type(value), "value")
