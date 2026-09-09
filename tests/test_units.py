"""Unit checks for the logic that has to be right for the loop to terminate."""
import json

import pytest

from hermes.config import Config, ConfigError, deep_merge, is_local_url
from hermes.evolve import Objective, parse_score
from hermes.implement import _protected, apply_change
from hermes.models import extract_json, strip_reasoning
from hermes.sandbox import Sandbox, SandboxError
from hermes.state import State, StateError, checkpoint, restore
from hermes.tickets import Board
from hermes.verify import scan_unsafe, verify_project
from hermes import workspace


# -- config ------------------------------------------------------------------

def test_local_url_gate():
    assert is_local_url("http://localhost:11434")
    assert is_local_url("http://192.168.1.10:8080/v1")
    assert not is_local_url("https://api.anthropic.com")
    assert not is_local_url("https://api.openai.com/v1")


def test_remote_model_is_rejected(tmp_path):
    path = tmp_path / "hermes.yaml"
    path.write_text("models:\n  default:\n    backend: openai\n"
                    "    base_url: https://api.openai.com/v1\n    model: gpt-4\n")
    with pytest.raises(ConfigError, match="not a local endpoint"):
        Config.load(path).model_for("coding")


def test_remote_allowed_only_with_explicit_optin(tmp_path):
    path = tmp_path / "hermes.yaml"
    path.write_text("models:\n  default:\n    backend: openai\n"
                    "    base_url: https://inference.example.com/v1\n"
                    "    model: m\n    allow_remote: true\n")
    assert Config.load(path).model_for("coding").model == "m"


def test_deep_merge_keeps_untouched_defaults():
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}}


# -- model output parsing ----------------------------------------------------

def test_extract_json_survives_reasoning_and_fences():
    text = '<think>let me see</think>\nSure:\n```json\n{"a": [1, 2], "b": "}"}\n```\nok'
    assert extract_json(text) == {"a": [1, 2], "b": "}"}


def test_extract_json_finds_bare_object_in_prose():
    assert extract_json('here it is {"x": {"y": 1}} thanks') == {"x": {"y": 1}}


def test_extract_json_raises_when_absent():
    with pytest.raises(ValueError):
        extract_json("no structured output at all")


def test_strip_reasoning():
    assert strip_reasoning("<think>a</think>b") == "b"


# -- state -------------------------------------------------------------------

def test_illegal_transition_rejected():
    state = State(idea="x")
    with pytest.raises(StateError):
        state.transition("RELEASE")


def test_state_roundtrip_and_checkpoint(tmp_path):
    state = workspace.create(tmp_path, "build a thing")
    state.transition("RESEARCH")
    state.save(tmp_path)
    assert State.load(tmp_path).phase == "RESEARCH"

    snapshot = checkpoint(tmp_path, "after-research")
    state.transition("SPECIFICATION")
    state.save(tmp_path)
    restore(tmp_path, snapshot)
    assert State.load(tmp_path).phase == "RESEARCH"


# -- tickets -----------------------------------------------------------------

def test_dependencies_gate_readiness(tmp_path):
    board = Board(tmp_path)
    first = board.add(title="foundation")
    second = board.add(title="feature", dependencies=[first.id])
    board.unblock()
    assert board.next_ticket().id == first.id
    board.move(first.id, "completed")
    assert board.next_ticket().id == second.id


def test_cycles_detected(tmp_path):
    board = Board(tmp_path)
    a = board.add(title="a")
    b = board.add(title="b", dependencies=[a.id])
    a.dependencies = [b.id]
    assert board.cycles()


def test_scoring_prefers_unblocking_work(tmp_path):
    board = Board(tmp_path)
    hub = board.add(title="hub", priority=2)
    leaf = board.add(title="leaf", priority=2)
    for _ in range(3):
        board.add(title="dependent", dependencies=[hub.id])
    assert board.score(hub) > board.score(leaf)


def test_repeated_failures_deprioritise(tmp_path):
    board = Board(tmp_path)
    ticket = board.add(title="flaky", priority=1)
    before = board.score(ticket)
    ticket.attempts = 3
    assert board.score(ticket) < before


def test_board_renders_markdown(tmp_path):
    board = Board(tmp_path)
    ticket = board.add(title="write parser", acceptance=["parses input"])
    board.save()
    files = list((tmp_path / "tickets" / "backlog").glob("*.md"))
    assert files and ticket.id in files[0].read_text()


# -- sandbox and file writes -------------------------------------------------

def test_sandbox_blocks_escapes(tmp_path):
    sandbox = Sandbox(tmp_path, {"runtime": "local", "timeout": 10})
    with pytest.raises(SandboxError):
        sandbox.write("../escape.txt", "x")
    sandbox.write("src/ok.txt", "x")
    assert (tmp_path / "src" / "ok.txt").read_text() == "x"


def test_local_sandbox_runs_and_times_out(tmp_path):
    sandbox = Sandbox(tmp_path, {"runtime": "local", "timeout": 2})
    assert sandbox.run("echo hello").stdout.strip() == "hello"
    assert sandbox.run("sleep 5", timeout=1).timed_out


def test_sandbox_strips_secrets_from_local_env(tmp_path):
    sandbox = Sandbox(tmp_path, {"runtime": "local", "timeout": 5})
    import os
    os.environ["MY_SECRET_TOKEN"] = "sensitive"
    try:
        assert "MY_SECRET_TOKEN" not in sandbox._local_env()
    finally:
        os.environ.pop("MY_SECRET_TOKEN")


def test_protected_paths():
    assert _protected(".hermes/state/state.json")
    assert _protected("/etc/passwd")
    assert _protected("a/../../b")
    assert not _protected("src/main.py")


def test_apply_change_rejects_and_writes(tmp_path):
    sandbox = Sandbox(tmp_path, {"runtime": "local", "timeout": 5})
    change = apply_change(sandbox, {
        "summary": "add module",
        "files": [{"path": "src/a.py", "content": "x = 1\n"},
                  {"path": "../evil.py", "content": "bad"},
                  {"path": ".hermes/state/state.json", "content": "bad"}],
    })
    assert change.written == ["src/a.py"]
    assert len(change.rejected) == 2
    assert not (tmp_path.parent / "evil.py").exists()


# -- verification ------------------------------------------------------------

def test_security_scan_flags_real_problems(tmp_path):
    (tmp_path / "bad.py").write_text(
        'import os\nos.system("rm " + name)\nKEY = "AKIAABCDEFGHIJKLMNOP"\n')
    (tmp_path / "good.py").write_text("import subprocess\nsubprocess.run(['ls'])\n")
    hits = scan_unsafe(tmp_path)
    assert any("bad.py" in h for h in hits)
    assert not any("good.py" in h for h in hits)


# -- evolution ---------------------------------------------------------------

def test_score_line_wins_over_stray_numbers():
    assert parse_score("running 5 iterations\nSCORE: 1.25\n") == 1.25


def test_objective_direction():
    minimise = Objective("time", "minimize")
    maximise = Objective("throughput", "maximize")
    assert minimise.better(1.0, 2.0) and not minimise.better(3.0, 2.0)
    assert maximise.better(3.0, 2.0) and not maximise.better(1.0, 2.0)
    assert not minimise.better(None, 2.0)


# -- workspace ---------------------------------------------------------------

def test_workspace_layout(tmp_path):
    workspace.create(tmp_path, "an idea")
    for name in ("CONTEXT.md", "AGENTS.md", "SPEC.md", "PLAN.md", ".gitignore"):
        assert (tmp_path / name).is_file()
    for directory in ("src", "tests", "docs/decisions", "tickets/backlog",
                      ".hermes/state"):
        assert (tmp_path / directory).is_dir()


def test_decision_records_increment(tmp_path):
    workspace.create(tmp_path, "idea")
    workspace.write_decision(tmp_path, 1, "Use SQLite", "c", "d", "q")
    assert workspace.next_decision_number(tmp_path) == 2


def test_docker_install_gets_network_then_loses_it(shared_tmp):
    """Setup installs must persist in the container and leave it offline again."""
    import shutil
    if not shutil.which("docker"):
        pytest.skip("docker unavailable")
    tmp_path = shared_tmp
    sandbox = Sandbox(tmp_path, {"runtime": "docker", "timeout": 300,
                                 "image": "python:3.11-slim"})
    with sandbox:
        if sandbox.runtime != "docker":
            pytest.skip("docker present but unusable")
        assert sandbox.install("pip install --quiet pytest").ok
        assert sandbox.run("python -c 'import pytest'").ok
        assert not sandbox.run(
            "python -c \"import urllib.request as u; u.urlopen('http://example.com', timeout=4)\""
        ).ok


def test_missing_tool_is_installed_and_recorded(shared_tmp):
    """A gate failing with exit 127 is an absent tool, not a code defect."""
    import shutil
    if not shutil.which("docker"):
        pytest.skip("docker unavailable")
    tmp_path = shared_tmp
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    plan = {"commands": {"setup": "true", "build": "true", "test": "pytest tests -q"}}
    sandbox = Sandbox(tmp_path, {"runtime": "docker", "timeout": 300,
                                 "image": "python:3.11-slim"})
    with sandbox:
        if sandbox.runtime != "docker":
            pytest.skip("docker present but unusable")
        verification = verify_project(sandbox, plan)
    assert verification.passed, verification.report()
    assert sandbox.installed == ["pytest"]
    assert plan["commands"]["setup"] == "pip install pytest"


def test_src_layout_is_importable(tmp_path):
    """Hermes mandates src/, so code there must import without packaging."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "thing.py").write_text("VALUE = 7\n")
    sandbox = Sandbox(tmp_path, {"runtime": "local", "timeout": 30})
    assert "7" in sandbox.run("python3 -c 'import thing; print(thing.VALUE)'").stdout
