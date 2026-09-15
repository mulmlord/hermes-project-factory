"""End-to-end check that the autonomous loop reaches COMPLETED unaided.

A stub router stands in for the local model so the whole state machine --
intake, planning, ticket execution, verification, review, documentation and
clean-room release -- runs deterministically without inference.
"""
import json

import pytest

from hermes import workspace
from hermes.config import Config
from hermes.factory import Factory
from hermes.journal import Journal

CALC = '''\
"""Tiny calculator."""


def add(a, b):
    return a + b


def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b
'''

TEST = '''\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from calc import add, divide

assert add(2, 3) == 5
assert divide(6, 3) == 2
try:
    divide(1, 0)
except ZeroDivisionError:
    pass
else:
    raise AssertionError("expected ZeroDivisionError")
print("ok")
'''

SPEC = {
    "name": "tiny-calc",
    "objective": "A minimal arithmetic library.",
    "deliverables": ["src/calc.py", "tests/test_calc.py"],
    "platforms": ["linux"],
    "technologies": ["Python"],
    "functional_requirements": ["addition", "division with a zero guard"],
    "nonfunctional_requirements": ["no third-party dependencies"],
    "constraints": ["standard library only"],
    "risks": ["none"],
    "unknowns": [],
    "success_criteria": ["tests pass"],
    "assumptions": ["Python 3 is available"],
    "optimization_objective": {"metric": "", "direction": "minimize"},
}

PLAN = {
    "approach": "Write the module, then the tests.",
    "architecture": "One module under src/, one test script under tests/.",
    "components": [{"name": "calc", "responsibility": "arithmetic", "path": "src/calc.py"}],
    "milestones": [{"name": "working library", "description": "add and divide",
                    "exit_criteria": "tests pass"}],
    "commands": {"setup": "true", "build": "python3 -c 'import sys'",
                 "test": "python3 tests/test_calc.py", "lint": "", "benchmark": ""},
    "layout": ["src/calc.py", "tests/test_calc.py"],
    "decisions": [{"title": "No dependencies", "decision": "Standard library only",
                   "rationale": "arithmetic needs nothing else",
                   "alternatives": "numpy", "consequences": "trivial install"}],
}

TICKETS = {"tickets": [
    {"key": "calc", "title": "Implement the calc module", "objective": "add and divide",
     "context": "", "kind": "implementation", "dependencies": [],
     "acceptance": ["add works", "divide guards zero"],
     "validation": ["python3 -c \"import sys; sys.path.insert(0,'src'); import calc\""],
     "artifacts": ["src/calc.py"], "priority": 1, "risk": 1, "cost": 1, "info_gain": 1},
    {"key": "tests", "title": "Add tests for calc", "objective": "cover both functions",
     "context": "", "kind": "test", "dependencies": ["calc"],
     "acceptance": ["tests pass"], "validation": ["python3 tests/test_calc.py"],
     "artifacts": ["tests/test_calc.py"], "priority": 1, "risk": 1, "cost": 1,
     "info_gain": 1},
]}


class StubRouter:
    """Answers each prompt by matching the marker text the caller sends."""

    def __init__(self):
        self.usage = []
        self.calls = []

    def describe(self, task):
        return "stub:model"

    def preflight(self):
        return ["stub ok"]

    def complete(self, task, system, prompt, **kw):
        return json.dumps(self._answer(prompt))

    def complete_json(self, task, system, prompt, **kw):
        return self._answer(prompt)

    def _answer(self, prompt):
        self.calls.append(prompt[:60])
        if "Produce a complete specification" in prompt:
            return SPEC
        if "Decide what genuinely requires research" in prompt:
            return {"needed": False, "topics": []}
        if "Design the system and define how it is built" in prompt:
            return PLAN
        if "Break this into at most" in prompt:
            return TICKETS
        if "Implement this ticket completely" in prompt:
            if "KIND: test" in prompt:
                return {"summary": "add tests",
                        "files": [{"path": "tests/test_calc.py", "content": TEST}]}
            return {"summary": "add calc module",
                    "files": [{"path": "src/calc.py", "content": CALC}]}
        if "Review this change" in prompt:
            return {"approved": True, "meets_acceptance": True, "blocking": [],
                    "advisory": [], "security": [], "reason": "correct and adequate"}
        if "Write the project README" in prompt:
            return {"purpose": "A minimal arithmetic library with add and divide.",
                    "features": "- addition\n- division with a zero guard",
                    "architecture": "`src/calc.py` holds the functions; "
                                    "`tests/test_calc.py` exercises them.",
                    "requirements": "Python 3.10 or newer. No dependencies.",
                    "installation": "Clone the repository. Nothing to install.",
                    "configuration": "None.",
                    "usage": "```python\nfrom calc import add\nadd(2, 3)\n```",
                    "examples": "```console\n$ python3 tests/test_calc.py\nok\n```",
                    "testing": "```console\n$ python3 tests/test_calc.py\n```",
                    "benchmarks": "None.",
                    "limitations": "Only addition and division are implemented.",
                    "license": "MIT"}
        if "Document the architecture as built" in prompt:
            return {"overview": "One module, one test script.",
                    "components": [{"name": "calc", "path": "src/calc.py",
                                    "responsibility": "arithmetic",
                                    "interfaces": "add, divide"}],
                    "data_flow": "Callers import calc directly.",
                    "key_decisions": ["standard library only"],
                    "extension_points": ["add more operations to calc.py"]}
        if "Which existing source files dominate" in prompt:
            return {"files": []}
        if "Diagnose the root cause" in prompt:
            return {"diagnosis": "unexpected", "classification": "implementation",
                    "action": "repair", "instructions": "fix it", "files": []}
        return {}


@pytest.fixture
def factory(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    workspace.create(root, "A minimal arithmetic library")
    config = Config.load()
    config.data["sandbox"]["runtime"] = "local"
    config.data["sandbox"]["timeout"] = 60
    config.data["github"]["enabled"] = False
    built = Factory(root, config, quiet=True)
    built.router = StubRouter()
    from hermes.agents import build as build_agents
    built.agents = build_agents(built.router, built.log)
    return built


def test_loop_reaches_completed(factory):
    state = factory.run(max_iterations=40)

    assert state.phase == "COMPLETED", (state.phase, state.notes)
    assert (factory.root / "src" / "calc.py").is_file()
    assert (factory.root / "tests" / "test_calc.py").is_file()

    board = factory.board
    assert board.progress() == 1.0
    assert not board.by_status("ready", "active", "blocked")

    readme = (factory.root / "README.md").read_text()
    assert "## Testing" in readme and len(readme) > 400

    criteria = state.metrics["completion"]
    assert all(criteria.values()), criteria

    # The workspace documents itself without Hermes internals.
    assert "tiny-calc" in (factory.root / "SPEC.md").read_text()
    assert (factory.root / "docs" / "architecture" / "overview.md").is_file()
    assert list((factory.root / "docs" / "decisions").glob("*.md"))

    # Every agent's work is recorded where the next agent will read it.
    journal = Journal(factory.root)
    assert {"src/calc.py", "tests/test_calc.py"} <= set(journal.owners())
    assert all(e.commit for e in journal.entries if e.outcome == "completed")
    assert (factory.root / "docs" / "generated" / "worklog.md").is_file()

    # Git history records the milestones.
    subjects = factory.repo.log_lines(30)
    assert any("feat:" in s for s in subjects), subjects
    assert any("docs:" in s for s in subjects), subjects


def test_state_survives_interruption(factory):
    factory.run(max_iterations=4)
    from hermes.state import State
    resumed = State.load(factory.root)
    assert resumed.phase != "INTAKE"
    assert resumed.iteration >= 1

    from hermes.factory import Factory
    second = Factory(factory.root, factory.config, quiet=True)
    second.router = factory.router
    from hermes.agents import build as build_agents
    second.agents = build_agents(second.router, second.log)
    assert second.run(max_iterations=40).phase == "COMPLETED"


def test_prefetch_refuses_a_ticket_that_writes_the_same_file(factory):
    """Generating ahead is only safe while the two tickets cannot collide."""
    board = factory.board
    current = board.add(title="writer", artifacts=["src/calc.py"], status="active")
    clash = board.add(title="same file", artifacts=["src/calc.py"], status="ready",
                      priority=1)
    assert factory._independent_ticket(current) is None

    clash.priority = 5
    other = board.add(title="elsewhere", artifacts=["src/other.py"], status="ready",
                      priority=1)
    assert factory._independent_ticket(current).id == other.id


def test_prefetched_work_is_reused_exactly_once(factory):
    """Generated ahead of time, handed to the ticket it was generated for."""
    board = factory.board
    current = board.add(title="writer", artifacts=["src/calc.py"], status="active")
    nxt = board.add(title="elsewhere", artifacts=["src/other.py"], status="ready",
                    priority=1)

    factory._prefetch_next(current)
    assert factory._prefetch[0] == nxt.id
    payload = factory._take_prefetch(nxt)
    assert payload["files"][0]["path"] == "src/calc.py"  # the stub's answer
    assert factory._take_prefetch(nxt) is None           # consumed, not reusable

    # Prefetched for one ticket, never handed to a different one.
    factory._prefetch_next(current)
    assert factory._take_prefetch(current) is None
    factory._stop_prefetch()


def test_failed_gates_send_ticket_back(tmp_path, monkeypatch):
    """A ticket whose validation fails must not be marked completed."""
    root = tmp_path / "proj"
    root.mkdir()
    workspace.create(root, "broken project")
    config = Config.load()
    config.data["sandbox"]["runtime"] = "local"
    config.data["execution"]["max_ticket_attempts"] = 1

    class Broken(StubRouter):
        def _answer(self, prompt):
            if "Implement this ticket completely" in prompt:
                return {"summary": "broken", "files": [
                    {"path": "src/calc.py", "content": "def add(a, b):\n    return a - b\n"}]}
            return super()._answer(prompt)

    built = Factory(root, config, quiet=True)
    built.router = Broken()
    from hermes.agents import build as build_agents
    built.agents = build_agents(built.router, built.log)
    built.run(max_iterations=25)

    calc = built.board.tickets
    assert any(t.status in ("abandoned", "blocked", "ready")
               for t in calc.values()), built.board.summary()
    assert built.state.phase != "COMPLETED" or not all(
        built.state.metrics.get("completion", {}).values())
