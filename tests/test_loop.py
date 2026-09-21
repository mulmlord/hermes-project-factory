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
from hermes.models import Router

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

    def allowance(self, task):
        # A real 16k context, so a prompt the loop builds too large fails here
        # exactly as it would against the configured server.
        return Router(Config.load()).allowance(task)

    def preflight(self):
        return ["stub ok"]

    def routed_tasks(self):
        # No endpoint at all, so the loop starts no inference server.
        return []

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


def test_a_lookup_before_the_code_does_not_derail_the_loop(factory):
    """The whole pipeline with an implementer that looks first: still COMPLETED.

    The stub answers from marker text, so it cannot ask a question it needs the
    answer to. What this checks is the mechanism around it: the lookup is
    served, its result reaches the next prompt, and the ticket finishes.
    """
    router, prompts, looked = factory.router, [], []
    answer = router._answer

    def looks_first(prompt):
        prompts.append(prompt)
        if "Implement this ticket completely" in prompt and not looked:
            looked.append(prompt)
            return {"lookup": [{"tool": "list_dir", "path": "."},
                               {"tool": "grep", "pattern": "def add"}]}
        return answer(prompt)

    router._answer = looks_first
    state = factory.run(max_iterations=40)

    assert state.phase == "COMPLETED", (state.phase, state.notes)
    served = [p for p in prompts if "WHAT YOU LOOKED UP" in p]
    assert served and "SPEC.md" in served[0]      # the listing it asked for
    assert (factory.root / "src" / "calc.py").is_file()


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


def test_the_board_is_published_while_there_is_still_something_to_watch(factory,
                                                                       monkeypatch):
    """A repository that arrives at release is a repository nobody can observe.

    The issue projection cannot act before a remote exists, so the repository
    is created where the board first exists. The code follows the first ticket
    that passes its gates, so the first thing anyone sees is not a scaffold.
    """
    from hermes.git_ops import GitHub

    events = []

    def created(self, root, name, description):
        events.append(("create", factory.board.progress()))
        return "https://github.com/o/r"

    def pushed(self, root):
        events.append(("push", factory.board.progress()))
        return True

    monkeypatch.setattr(GitHub, "create", created)
    monkeypatch.setattr(GitHub, "push", pushed)
    monkeypatch.setattr(GitHub, "release", lambda self, *a, **k: None)
    factory.config.data["github"]["enabled"] = True

    assert factory.run(max_iterations=40).phase == "COMPLETED"

    assert events, "the repository was never created"
    # Created while the whole board was still outstanding, not after the fact.
    assert events[0] == ("create", 0.0)
    first_push = next(p for name, p in events if name == "push")
    assert 0.0 < first_push < 1.0, "pushed a scaffold, or pushed only at the end"


def test_a_project_without_github_publishes_nothing(factory, monkeypatch):
    """Off must mean the gh CLI is never invoked at all, not merely never pushed.

    Asserted at the process boundary on purpose: stubbing the publishing
    methods themselves would stub out the guard that is under test.
    """
    import subprocess

    from hermes.git_ops import GitHub

    calls = []

    def never(self, *args, cwd=None):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, "", "")

    monkeypatch.setattr(GitHub, "_gh", never)
    assert factory.run(max_iterations=40).phase == "COMPLETED"
    assert not calls


def test_networked_research_records_what_becomes_of_retrieved_pages(factory):
    """A project that reaches the network says, in itself, what it keeps."""
    factory.config.data["research"]["network"] = True
    assert factory._librarian() is not None

    records = list((factory.root / "docs" / "decisions").glob("*retrieved-*.md"))
    assert len(records) == 1
    text = records[0].read_text()
    assert "gitignored" in text and "docs/research/" in text

    factory._librarian()  # a second round must not restate the same decision
    assert len(list((factory.root / "docs" / "decisions").glob("*retrieved-*.md"))) == 1


def test_research_without_a_network_budget_takes_no_such_decision(factory):
    """A project that never fetched is not handed a decision it never took."""
    assert factory._librarian() is None
    assert not list((factory.root / "docs" / "decisions").glob("*retrieved-*.md"))


def test_every_log_record_names_the_work_that_produced_it(factory):
    """A log nobody can correlate cannot answer why one attempt failed."""
    state = factory.run(max_iterations=40)
    records = [json.loads(line)
               for path in sorted((factory.root / ".hermes" / "logs").glob("*.jsonl"))
               for line in path.read_text().splitlines()]

    run = state.metrics["run"]
    start = next(i for i, r in enumerate(records) if r["event"] == "factory.start")
    assert run and all(r.get("run") == run for r in records[start:])

    attempt = next(r for r in records if r["event"] == "ticket.attempt")
    assert attempt["phase"] == "IMPLEMENTATION" and attempt["attempt"] == 1
    assert attempt["ticket"] in factory.board.tickets

    # implement.py knows nothing about phases or attempts; the scope does.
    applied = next(r for r in records if r["event"] == "implement.applied")
    assert applied["ticket"] in factory.board.tickets
    assert applied["phase"] == "IMPLEMENTATION" and applied["attempt"] >= 1


def test_a_second_run_is_distinguishable_in_the_same_log_file(factory):
    """A resumed project appends to the log its earlier run wrote."""
    first = factory.run(max_iterations=2).metrics["run"]
    second = factory.run(max_iterations=40).metrics["run"]
    assert first != second

    records = [json.loads(line)
               for path in sorted((factory.root / ".hermes" / "logs").glob("*.jsonl"))
               for line in path.read_text().splitlines()]
    assert {first, second} <= {r.get("run") for r in records}


def test_a_wall_clock_deadline_stops_the_loop_without_damaging_it(factory):
    """Iterations bound how many phases run, not how long one takes.

    A single stuck command is otherwise bounded only by the sandbox timeout,
    which is what lets one bad idea eat a whole suite run. Stopping must leave
    the project resumable: a deadline is a pause, not a failure.
    """
    import time

    state = factory.run(max_iterations=40, deadline=time.time() - 1)
    assert state.phase != "COMPLETED"
    assert state.metrics["deadline_reached"] is True
    assert any("deadline" in note for note in state.notes)

    from hermes.state import State
    assert State.load(factory.root).phase == state.phase   # the stop is persisted

    factory.state.metrics.pop("deadline_reached")
    assert factory.run(max_iterations=40).phase == "COMPLETED"


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


def test_a_regressing_review_retry_keeps_the_version_that_passed(tmp_path):
    """A blocking review must not cost the factory the code that already worked."""
    root = tmp_path / "proj"
    root.mkdir()
    workspace.create(root, "A minimal arithmetic library")
    config = Config.load()
    config.data["sandbox"]["runtime"] = "local"
    config.data["sandbox"]["timeout"] = 60
    config.data["github"]["enabled"] = False
    config.data["execution"]["max_ticket_attempts"] = 3

    class Picky(StubRouter):
        """Blocks the first change it reviews, then breaks the module fixing it."""

        def __init__(self):
            super().__init__()
            self.prompts = []
            self.blocked = False

        def _answer(self, prompt):
            if "Implement this ticket completely" in prompt:
                self.prompts.append(prompt)
                if ("KIND: implementation" in prompt and "REPAIR INSTRUCTIONS" in prompt
                        and "PREVIOUS ATTEMPT FAILED" not in prompt):
                    # The rewrite that answers the reviewer and stops importing.
                    return {"summary": "guard the inputs", "files": [
                        {"path": "src/calc.py",
                         "content": "def add(a, b)\n    return a + b\n"}]}
            if "Review this change" in prompt and not self.blocked:
                self.blocked = True
                return {"approved": False, "meets_acceptance": False,
                        "blocking": ["add does not guard non-numeric input"],
                        "advisory": [], "security": [], "reason": "needs a guard"}
            return super()._answer(prompt)

    built = Factory(root, config, quiet=True)
    built.router = Picky()
    from hermes.agents import build as build_agents
    built.agents = build_agents(built.router, built.log)
    state = built.run(max_iterations=40)

    assert state.phase == "COMPLETED", (state.phase, state.notes)
    assert (root / "src" / "calc.py").read_text() == CALC
    repaired = [p for p in built.router.prompts if "PREVIOUS ATTEMPT FAILED" in p]
    assert repaired, "the regression was never reported back to the implementer"
    assert "broke gates that had already passed" in repaired[0]
    assert "add does not guard non-numeric input" in repaired[0]
    assert "return a + b" in repaired[0]  # the restored module, not the broken retry


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


def test_a_run_starts_the_model_server_and_stops_it_afterwards(factory, monkeypatch):
    """Nothing else brings the server up: the unit is installed but not enabled.

    So the wiring matters as much as the mechanism — a run that forgot to
    start it would fail preflight on a machine where everything is installed
    correctly, and one that forgot to stop it would leave the model in VRAM.
    """
    from types import SimpleNamespace

    calls = []
    up = {"now": False}

    def systemctl(args):
        calls.append(args[0])
        if args[0] == "cat":
            return 0, "[Unit]\n"
        if args[0] == "start":
            up["now"] = True
        if args[0] == "stop":
            up["now"] = False
        return 0, ""

    def available_models():
        if not up["now"]:
            raise RuntimeError("connection refused")
        return ["stub:model"]

    monkeypatch.setattr("hermes.server._systemctl", systemctl)
    monkeypatch.setattr(factory.router, "routed_tasks", lambda: ["coding"],
                        raising=False)
    monkeypatch.setattr(factory.router, "backend", lambda task: SimpleNamespace(
        cfg=SimpleNamespace(base_url="http://localhost:6767/v1"),
        available_models=available_models), raising=False)

    factory.run(max_iterations=1)
    assert calls == ["cat", "start", "stop"]
    assert not up["now"]


def test_one_page_budget_serves_the_whole_run(factory):
    """Research is entered once, then again by every failure triaged as research.

    Each of those built its own librarian, so `research.max_fetches` bounded a
    call rather than the run: ten escalations fetched ten times the configured
    pages and nothing counted them. One librarian for the run is what makes
    the configured number the number.
    """
    factory.config.data["research"]["network"] = True

    first = factory._librarian()
    first.spent = first.max_pages

    assert factory._librarian() is first
    assert factory._librarian().remaining == 0
