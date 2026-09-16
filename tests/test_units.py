"""Unit checks for the logic that has to be right for the loop to terminate."""
import json
import subprocess
import threading
import time

import pytest

from types import SimpleNamespace

from hermes.config import Config, ConfigError, deep_merge, is_local_url
from hermes.evolve import Bandit, Candidate, Evolution, Objective, parse_score
from hermes.implement import (Change, _protected, _race, apply_change, execute_ticket,
                              relevant_files, restore as restore_files,
                              snapshot as snapshot_files)
from hermes.git_ops import GitHub, Issues
from hermes.journal import Entry, Journal
from hermes.log import NULL
from hermes.models import ModelError, Router, extract_json, strip_reasoning
from hermes.openhands import OpenHands
from hermes.research import investigate
from hermes.resources import Governor
from hermes import harness
from hermes.sandbox import Sandbox, SandboxError
from hermes.sources import Librarian
from hermes.state import State, StateError, checkpoint, restore
from hermes.tickets import Board, split_validation
from hermes.verify import scan_unsafe, verify_project
from hermes import cli, workspace


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


def test_config_is_read_from_where_the_installer_writes_it(tmp_path, monkeypatch):
    """Both scripts write to $XDG_CONFIG_HOME/hermes/config.yaml when it is set.

    Reading only ~/.config would mean the installer records the served model's
    name somewhere Hermes never looks, and the factory would silently run on
    built-in defaults naming a model the server does not serve.
    """
    from pathlib import Path
    from hermes.config import user_config_paths

    cfg = tmp_path / "hermes" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("models:\n  default:\n    backend: vllm\n"
                   "    base_url: http://localhost:6767/v1\n    model: served-model\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)  # no hermes.yaml here, so the user config wins

    loaded = Config.load()
    assert loaded.path == cfg
    assert loaded.model_for("coding").model == "served-model"

    # Unset, the historical location is still the one used.
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert cfg not in user_config_paths()
    assert user_config_paths() == [Path.home() / ".config" / "hermes" / "config.yaml"]


def test_defaults_address_the_server_install_sh_starts():
    """install.sh serves vLLM on 6767; the built-in defaults must agree.

    Three files describe that endpoint — the defaults, the sample config and
    the installer's sed patch — and a project run with no hermes.yaml of its
    own reaches the server through these.
    """
    from hermes.config import DEFAULTS, SAMPLE_CONFIG
    from hermes.models import BACKENDS, OpenAICompatBackend

    cfg = Config(deep_merge(DEFAULTS, {})).model_for("coding")
    assert cfg.backend == "vllm"
    assert cfg.base_url == "http://localhost:6767/v1"
    assert is_local_url(cfg.base_url)
    # vLLM speaks the OpenAI protocol; nothing else would reach it.
    assert BACKENDS[cfg.backend] is OpenAICompatBackend
    assert OpenAICompatBackend(cfg)._base() == "http://localhost:6767/v1"

    # The installer rewrites these three keys in the sample by anchoring on
    # four leading spaces, so they have to stay exactly one level deep.
    for key in ("base_url", "model", "context"):
        assert f"\n    {key}: " in SAMPLE_CONFIG
    assert "6767" in SAMPLE_CONFIG


# -- openhands ---------------------------------------------------------------

def test_openhands_delegation_stays_headless_and_local(tmp_path, monkeypatch):
    """The invocation must match the CLI Hermes actually shells out to.

    Two of these assertions are load-bearing rather than cosmetic: there is no
    --directory flag, so passing one aborts the run, and without
    --override-with-envs the LLM_* variables are ignored and OpenHands falls
    back to a hosted endpoint, which Hermes is not allowed to reach.
    """
    model = SimpleNamespace(backend="ollama", model="qwen3:30b-a3b",
                            base_url="http://localhost:11434")
    agent = OpenHands({"enabled": True, "command": "openhands"}, model)
    ticket = SimpleNamespace(id="T001", title="tidy", kind="refactor",
                             objective="o", context="c", acceptance=[],
                             validation=[], artifacts=[])
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        seen["env"] = kw.get("env", {})
        seen["cwd"] = kw.get("cwd")
        return SimpleNamespace(returncode=0, stdout="done", stderr="")

    monkeypatch.setattr("hermes.openhands.subprocess.run", fake_run)
    assert agent.run_ticket(tmp_path, ticket, "CONTEXT")["ok"]

    assert "--headless" in seen["argv"]
    assert "--override-with-envs" in seen["argv"]
    assert "-t" in seen["argv"]
    assert "--directory" not in seen["argv"]
    assert seen["cwd"] == tmp_path  # the working directory is the cwd, not a flag

    env = seen["env"]
    assert env["LLM_MODEL"] == "ollama/qwen3:30b-a3b"
    assert is_local_url(env["LLM_BASE_URL"])
    assert not [k for k in env if "ANTHROPIC" in k.upper() or "OPENAI_API_KEY" in k]


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


def test_json_mode_downgrades_instead_of_failing(monkeypatch):
    """A server that rejects the JSON parameter must not fail the whole task."""
    router = Router(Config.load())
    backend = router.backend("coding")
    asked = []

    def fake(system, prompt, *, json_mode=False, **kw):
        asked.append(json_mode)
        if json_mode:
            raise ModelError("http://localhost:11434/api/chat: HTTP 400 bad format")
        return '{"ok": true}'

    monkeypatch.setattr(backend, "complete", fake)
    assert router.complete_json("coding", "sys", "prompt") == {"ok": True}
    assert asked == [True, False]
    assert backend.json_mode is False  # and it does not ask again


def test_a_server_fault_is_not_mistaken_for_a_missing_feature(monkeypatch):
    router = Router(Config.load())
    backend = router.backend("coding")
    monkeypatch.setattr(backend, "complete", lambda *a, **k: (_ for _ in ()).throw(
        ModelError("http://localhost:11434: HTTP 500 out of memory")))
    with pytest.raises(ModelError, match="500"):
        router.complete_json("coding", "sys", "prompt")
    assert backend.json_mode is True


def test_only_routed_models_are_preflighted():
    """A model no agent ever asks for must not be able to fail a run."""
    from hermes.agents import ROUTING
    tasks = Router(Config.load()).routed_tasks()
    assert set(tasks) == set(ROUTING.values())
    assert "simple" in tasks  # the summariser gives the small model real work


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


def test_longest_chain_beats_a_wide_fan_out(tmp_path):
    """A project finishes when its longest chain finishes, not its widest."""
    board = Board(tmp_path)
    chain = board.add(title="chain head", priority=2, cost=1)
    previous = chain
    for step in range(3):
        previous = board.add(title=f"step {step}", dependencies=[previous.id], cost=3)
    hub = board.add(title="hub", priority=2, cost=1)
    for _ in range(4):
        board.add(title="leaf", dependencies=[hub.id], cost=1)
    assert board.critical_path(chain) > board.critical_path(hub)
    assert board.score(chain) > board.score(hub)


def test_finished_work_cannot_extend_the_critical_path(tmp_path):
    board = Board(tmp_path)
    hub = board.add(title="hub", cost=1)
    dependent = board.add(title="dependent", dependencies=[hub.id], cost=5)
    before = board.critical_path(hub)
    board.move(dependent.id, "completed")
    assert board.critical_path(hub) < before


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


def test_edit_protocol_offers_edits_only_when_enabled():
    from hermes.agents import file_protocol
    assert '"edits"' in file_protocol(True)
    assert '"edits"' not in file_protocol(False)


def test_an_edit_replaces_a_unique_match(tmp_path):
    sandbox = Sandbox(tmp_path, {"runtime": "local", "timeout": 5})
    sandbox.write("src/a.py", "def add(a, b):\n    return a - b\n")
    change = apply_change(sandbox, {"summary": "fix the sign", "edits": [
        {"path": "src/a.py", "find": "return a - b", "replace": "return a + b"}]})
    assert change.written == ["src/a.py"] and not change.unmatched
    assert sandbox.read("src/a.py") == "def add(a, b):\n    return a + b\n"


def test_an_ambiguous_edit_is_refused_with_a_reason(tmp_path):
    """Guessing which identical anchor was meant is how working code is lost."""
    sandbox = Sandbox(tmp_path, {"runtime": "local", "timeout": 5})
    sandbox.write("src/a.py", "x = 1\nx = 1\n")
    change = apply_change(sandbox, {"edits": [
        {"path": "src/a.py", "find": "x = 1", "replace": "x = 2"}]})
    assert sandbox.read("src/a.py") == "x = 1\nx = 1\n"
    assert change.empty and "appears 2 times" in change.unmatched[0]


def test_an_edit_to_a_missing_file_asks_for_the_whole_file(tmp_path):
    sandbox = Sandbox(tmp_path, {"runtime": "local", "timeout": 5})
    change = apply_change(sandbox, {"edits": [
        {"path": "src/new.py", "find": "a", "replace": "b"}]})
    assert change.empty and "does not exist" in change.unmatched[0]


def test_an_edit_cannot_reach_a_protected_file(tmp_path):
    sandbox = Sandbox(tmp_path, {"runtime": "local", "timeout": 5})
    sandbox.write(".hermes/state/state.json", '{"phase": "INTAKE"}')
    change = apply_change(sandbox, {"edits": [
        {"path": ".hermes/state/state.json", "find": "INTAKE", "replace": "COMPLETED"}]})
    assert change.rejected == [".hermes/state/state.json"]
    assert "INTAKE" in sandbox.read(".hermes/state/state.json")


def test_a_losing_candidate_is_rolled_back(tmp_path):
    """The next sample must start from the workspace the first one saw."""
    sandbox = Sandbox(tmp_path, {"runtime": "local", "timeout": 5})
    sandbox.write("src/a.py", "original\n")
    payloads = [{"files": [{"path": "src/a.py", "content": "first\n"}]},
                {"files": [{"path": "src/b.py", "content": "second\n"}]}]
    seen = []

    def gate(change):
        seen.append(sandbox.read("src/a.py"))
        return SimpleNamespace(passed=len(seen) == 2, gates=[], report=lambda: "")

    change, verification = _race(sandbox, payloads, gate,
                                 SimpleNamespace(id="T001"), NULL)
    assert seen == ["first\n", "original\n"]  # the loser's write was undone
    assert verification.passed and change.written == ["src/b.py"]


def test_snapshot_restores_a_file_that_did_not_exist(tmp_path):
    sandbox = Sandbox(tmp_path, {"runtime": "local", "timeout": 5})
    saved = snapshot_files(sandbox, ["src/new.py"])
    sandbox.write("src/new.py", "written\n")
    restore_files(sandbox, saved)
    assert not (tmp_path / "src" / "new.py").exists()


#: One attempt each for the review cycle below: the change that clears the
#: gates, the rewrite that answers the reviewer by breaking them, and the one
#: that manages both.
PASSES = {"summary": "carry the value",
          "files": [{"path": "src/a.py", "content": "value = 'good'\n"}]}
REGRESSES = {"summary": "answer the reviewer",
             "files": [{"path": "src/a.py", "content": "value = 'broken'\n"},
                       {"path": "src/extra.py", "content": "helper = 1\n"}]}
SATISFIES = {"summary": "answer the reviewer without breaking anything",
             "files": [{"path": "src/a.py", "content": "value = 'good'  # guarded\n"}]}


def _review_cycle(tmp_path, replies):
    """Run one ticket whose first attempt passes the gates but is blocked in review.

    *replies* answers the implementer once per attempt. The reviewer approves
    only the guarded version, so the ticket can finish in no other way than by
    a change that clears the gates and the finding at the same time.
    """
    sandbox = Sandbox(tmp_path, {"runtime": "local", "timeout": 30})
    board = Board(tmp_path)
    ticket = board.add(title="carry a value", status="ready", artifacts=["src/a.py"],
                       validation=["grep -q good src/a.py"])
    state = State(idea="carry a value",
                  plan={"commands": {"build": "true", "test": "true"}})
    prompts = []

    def implement(prompt):
        prompts.append(prompt)
        return replies[min(len(prompts), len(replies)) - 1]

    def reviewer(_prompt):
        approved = "guarded" in sandbox.read("src/a.py")
        return {"approved": approved, "advisory": [], "security": [],
                "blocking": [] if approved else ["guard the empty input case"]}

    def triager(_prompt):
        return {"diagnosis": "the value is wrong", "classification": "implementation",
                "action": "repair", "instructions": "write the right value"}

    agents = {name: SimpleNamespace(ask_json=lambda prompt, _fn=fn, **kw: _fn(prompt))
              for name, fn in (("implementer", implement), ("tester", implement),
                               ("reviewer", reviewer), ("triager", triager))}
    outcome = execute_ticket(agents, sandbox, state, board, ticket, lambda: "src/a.py",
                             max_attempts=len(replies))
    return outcome, sandbox, prompts, ticket


def test_a_review_retry_that_regresses_a_gate_is_rolled_back(tmp_path):
    """Work that already passed must survive the attempt to answer the reviewer."""
    outcome, sandbox, prompts, _ = _review_cycle(tmp_path, [PASSES, REGRESSES, SATISFIES])

    assert outcome["status"] == "completed"
    assert "guarded" in sandbox.read("src/a.py")
    # The retry's second file went back with the first: answering a review is
    # not confined to the files the passing attempt happened to write.
    assert not (tmp_path / "src" / "extra.py").exists()
    # The next attempt was told both what it broke and what it was sent back for.
    assert "broke gates that had already passed" in prompts[2]
    assert "guard the empty input case" in prompts[2]


def test_a_restored_snapshot_does_not_satisfy_the_review(tmp_path):
    """Recovering the passing tree answers the reviewer's objection not at all."""
    outcome, sandbox, _prompts, ticket = _review_cycle(tmp_path, [PASSES, REGRESSES])

    assert outcome["status"] == "failed"
    assert sandbox.read("src/a.py") == "value = 'good'\n"  # restored, not the retry's
    assert ticket.status != "completed"


# -- work journal ------------------------------------------------------------

def _done(journal, ticket, summary, files):
    return journal.record(ticket, {
        "status": "completed",
        "change": Change(summary=summary, written=files)})


def test_journal_prefers_dependencies_over_recency(tmp_path):
    """Recent noise must not crowd out the ticket this one is built on."""
    board, journal = Board(tmp_path), Journal(tmp_path)
    base = board.add(title="config loader", artifacts=["src/config.py"])
    ticket = board.add(title="wire config in", dependencies=[base.id])
    _done(journal, base, "reads hermes.yaml", ["src/config.py"])
    for index in range(30):
        _done(journal, board.add(title=f"unrelated {index}"), "x", [f"src/n{index}.py"])

    block = journal.block(ticket, board, budget=400)
    assert base.id in block and "src/config.py" in block
    assert len(block) < 1200  # budget honoured, not the whole history
    assert journal.suggested_files(ticket, board)[0] == "src/config.py"


def test_journal_carries_dead_ends_to_the_next_agent(tmp_path):
    board, journal = Board(tmp_path), Journal(tmp_path)
    failed = board.add(title="thread pool scheduler")
    journal.record(failed, {"diagnosis": {"diagnosis": "deadlocked under load"}},
                   status="abandoned")
    block = journal.block(board.add(title="scheduler, second attempt"), board)
    assert "abandoned" in block and "deadlocked under load" in block


def test_journal_owner_index_names_the_newest_writer(tmp_path):
    board, journal = Board(tmp_path), Journal(tmp_path)
    _done(journal, board.add(title="first"), "created it", ["src/a.py"])
    second = board.add(title="second")
    _done(journal, second, "rewrote it", ["src/a.py"])
    assert journal.owners()["src/a.py"].ticket == second.id


def test_journal_keeps_only_the_latest_attempt_per_ticket(tmp_path):
    board, journal = Board(tmp_path), Journal(tmp_path)
    ticket = board.add(title="parser")
    journal.record(ticket, {"diagnosis": {"diagnosis": "wrong grammar"}}, status="failed")
    _done(journal, ticket, "fixed the grammar", ["src/parser.py"])
    latest = journal.latest()
    assert len(latest) == 1 and latest[0].outcome == "completed"


def test_journal_survives_a_torn_line(tmp_path):
    """A crash mid-append must not make the journal unreadable."""
    journal = Journal(tmp_path)
    journal.append(Entry(ticket="T001", title="ok", files=["src/a.py"]))
    with journal.path.open("a") as fh:
        fh.write('{"ticket": "T002", "tit')
    assert [e.ticket for e in Journal(tmp_path).entries] == ["T001"]


def test_journal_mirrors_to_a_readable_worklog(tmp_path):
    journal = Journal(tmp_path)
    journal.append(Entry(ticket="T001", title="config loader", summary="reads yaml"))
    text = (tmp_path / "docs" / "generated" / "worklog.md").read_text()
    assert "T001" in text and "reads yaml" in text


def test_relevant_files_reads_journal_hints(tmp_path):
    """Hints point at files that exist; ticket artifacts are only a guess."""
    sandbox = Sandbox(tmp_path, {"runtime": "local", "timeout": 5})
    sandbox.write("src/config.py", "VALUE = 1\n")
    ticket = Board(tmp_path).add(title="use the config", artifacts=["src/never_built.py"])
    text = relevant_files(sandbox, ticket, {}, ["src/config.py"])
    assert "VALUE = 1" in text


# -- resource governor -------------------------------------------------------

def _governor(cfg=None, **probe):
    """A governor over a fixed probe, so the host running the tests cannot
    decide whether they pass."""
    return Governor(cfg or {}, NULL, probe=lambda: probe)


def test_a_full_gpu_defers_the_prefetch():
    governor = _governor({"reserve_vram_mb": 1024}, gpus=[{"vram_free_mb": "300"}])
    assert not governor.admit_inference()
    assert "VRAM" in governor.last_refusal


def test_headroom_admits_a_second_generation():
    governor = _governor({"reserve_vram_mb": 1024}, gpus=[{"vram_free_mb": "7000"}])
    assert governor.admit_inference()


def test_a_benchmark_under_measurement_blocks_concurrent_generation():
    """Evolution selects on timings: a shared GPU measures contention instead."""
    governor = _governor(gpus=[{"vram_free_mb": "7000"}])
    with governor.measuring():
        assert not governor.admit_inference()
        assert "benchmark" in governor.last_refusal
    assert governor.admit_inference()


def test_the_concurrency_limit_is_honoured():
    governor = _governor({"max_concurrent_inference": 1},
                         gpus=[{"vram_free_mb": "7000"}])
    with governor.inference():
        assert not governor.admit_inference()
    assert governor.admit_inference()


def test_a_host_without_a_gpu_is_judged_on_free_ram():
    governor = _governor({"reserve_ram_gb": 8}, cpus=4, ram_free_gb=2.0)
    assert not governor.admit_inference()
    assert "RAM" in governor.last_refusal


def test_an_unreadable_probe_still_admits():
    """A machine with no nvidia-smi must not become one that never prefetches."""
    def boom():
        raise OSError("no nvidia-smi here")

    assert Governor({}, NULL, probe=boom).admit_inference()


def test_measuring_waits_for_inference_to_drain():
    governor = _governor({"measure_drain_seconds": 5},
                         gpus=[{"vram_free_mb": "7000"}])
    released = []

    def hold():
        with governor.inference():
            time.sleep(0.2)
            released.append(time.time())  # recorded before the slot is given up

    worker = threading.Thread(target=hold)
    worker.start()
    while not governor.status()["inflight"]:
        time.sleep(0.01)
    with governor.measuring():
        entered = time.time()
    worker.join()
    assert released and entered > released[0]


def test_a_stuck_model_call_cannot_block_a_benchmark_forever():
    """A bounded wait: a noisy measurement beats a loop that never measures."""
    governor = _governor({"measure_drain_seconds": 0.1},
                         gpus=[{"vram_free_mb": "7000"}])
    started = time.time()
    with governor.inference():
        with governor.measuring():
            pass
    assert time.time() - started < 3


def test_the_router_registers_every_call_with_the_governor():
    """The governor can only judge a second call if it was told about the first."""
    governor = _governor(gpus=[{"vram_free_mb": "7000"}])
    config = Config({"models": {"default": {"backend": "ollama", "model": "m"}},
                     "routing": {"coding": "default"}})
    router = Router(config, NULL, governor)
    seen = {}

    class Stub:
        cfg = SimpleNamespace(model="m")

        def complete(self, system, prompt, **kw):
            seen["inflight"] = governor.status()["inflight"]
            return "ok"

    router._cache["coding"] = Stub()
    router.complete("coding", "sys", "prompt")
    assert seen["inflight"] == 1
    assert governor.status()["inflight"] == 0


# -- networked research ------------------------------------------------------

GUIDE = ("<html><head><title>Ring Buffer Guide</title>"
         "<script>var noise = 'script text must not be quoted back';</script>"
         "</head><body><h1>Ring buffers</h1><p>A power-of-two capacity lets the "
         "index wrap with a bitmask instead of a modulo.</p></body></html>")


@pytest.fixture
def served():
    """A real HTTP server on loopback, so a fetch is an actual fetch.

    Its document root sits outside every project directory: a test that scans
    a project must find the retrieved copy, never the server's own files.
    """
    import functools
    import http.server
    import tempfile

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    with tempfile.TemporaryDirectory() as pages:
        for name, body in (
                ("guide.html", GUIDE),
                ("second.html", "<html><body>another page</body></html>"),
                ("setup.py", 'import os\nos.system("rm -rf " + target)\n')):
            with open(f"{pages}/{name}", "w") as handle:
                handle.write(body)
        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(Quiet, directory=pages))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            server.server_close()


def _librarian(root, **cfg):
    """A librarian over the local sandbox runtime, which needs no docker."""
    sandbox = Sandbox(root, {"runtime": "local", "timeout": 60}, NULL)
    return Librarian(sandbox, root, cfg, NULL)


class _Researcher:
    """Answers the three research prompts, citing whatever it was handed."""

    def __init__(self, url):
        self.url = url
        self.saw_material = False

    def ask_json(self, prompt, **kw):
        if "Step 1 of the Feynman method" in prompt:
            return {"simple_explanation": "how a ring buffer wraps its index",
                    "what_we_know": [],
                    "unknowns": [{"question": "How should the index wrap?",
                                  "why_it_matters": "it decides the per-op cost"}]}
        if "Name at most" in prompt:
            return {"sources": [{"url": self.url, "why": "the reference guide"}]}
        if "QUESTION:" in prompt:
            self.saw_material = "power-of-two capacity" in prompt
            return {"answer": "Mask the index.", "confidence": "verified",
                    "sources": ["[1]", "https://invented.example/never-fetched"],
                    "implications": ["choose a power-of-two capacity"]}
        return {"conclusions": ["mask, do not modulo"], "decisions": [],
                "hypotheses": [], "tickets": [], "remaining_unknowns": []}


def test_a_retrieved_page_is_stored_as_evidence_and_reduced_to_text(tmp_path, served):
    workspace.create(tmp_path, "a ring buffer")
    librarian = _librarian(tmp_path, allow_private=True)

    source = librarian.fetch(f"{served}/guide.html", why="how the index wraps")

    assert source is not None, librarian.refusals
    assert source.path.startswith("research/sources/")
    assert (tmp_path / source.path).read_bytes()      # the bytes as they arrived
    assert source.title == "Ring Buffer Guide"
    text = librarian.text(source, 4000)
    assert "power-of-two capacity" in text
    assert "must not be quoted back" not in text      # script stripped, not fed back


def test_a_private_address_is_refused_unless_deliberately_allowed(tmp_path, served):
    """The model names the URL; it does not get to name the operator's LAN."""
    workspace.create(tmp_path, "a ring buffer")
    librarian = _librarian(tmp_path)                  # allow_private defaults off

    assert librarian.fetch(f"{served}/guide.html") is None
    assert "non-public" in librarian.refusals[0]["reason"]
    assert not list((tmp_path / "research" / "sources").glob("*"))


def test_a_non_http_url_never_reaches_the_sandbox(tmp_path):
    workspace.create(tmp_path, "a ring buffer")
    librarian = _librarian(tmp_path, allow_private=True)
    librarian.sandbox = None                          # running anything would raise

    assert librarian.fetch("file:///etc/passwd") is None
    assert librarian.spent == 0
    assert "scheme" in librarian.refusals[0]["reason"]


def test_the_page_budget_bounds_one_round(tmp_path, served):
    workspace.create(tmp_path, "a ring buffer")
    librarian = _librarian(tmp_path, allow_private=True, max_fetches=1)

    found = librarian.gather([{"url": f"{served}/guide.html"},
                              {"url": f"{served}/second.html"}])

    assert len(found) == 1 and librarian.remaining == 0


def test_the_same_page_is_retrieved_once(tmp_path, served):
    workspace.create(tmp_path, "a ring buffer")
    librarian = _librarian(tmp_path, allow_private=True)
    url = f"{served}/guide.html"

    assert librarian.fetch(url) is librarian.fetch(url)
    assert librarian.spent == 1


def test_retrieved_material_cannot_fail_the_security_scan(tmp_path, served):
    """A fetched page is evidence, not code, and is never stored as source."""
    workspace.create(tmp_path, "a ring buffer")
    librarian = _librarian(tmp_path, allow_private=True)

    source = librarian.fetch(f"{served}/setup.py")   # unsafe patterns inside

    assert source is not None, librarian.refusals
    assert not source.path.endswith(".py")
    assert not scan_unsafe(tmp_path)


def test_research_answers_from_what_it_retrieved_and_cites_it(tmp_path, served):
    workspace.create(tmp_path, "a ring buffer")
    url = f"{served}/guide.html"
    researcher = _Researcher(url)

    payload = investigate({"researcher": researcher}, tmp_path, "ring buffers",
                          "CONTEXT", librarian=_librarian(tmp_path, allow_private=True))

    assert researcher.saw_material, "the retrieved text never reached the prompt"
    # "[1]" resolves to the page that was actually fetched; the invented URL goes.
    assert payload["findings"][0]["sources"] == [url]
    assert payload["sources"][0]["path"].startswith("research/sources/")

    notes = next((tmp_path / "docs" / "research").glob("*.md")).read_text()
    assert url in notes and "Sources retrieved" in notes


def test_research_without_a_network_budget_fetches_nothing(tmp_path):
    """The default: findings come from the model, and say so."""
    assert Config.load().get("research.network") is False
    workspace.create(tmp_path, "a ring buffer")
    researcher = _Researcher("http://never.example/")

    payload = investigate({"researcher": researcher}, tmp_path, "ring buffers",
                          "CONTEXT")

    assert payload["sources"] == []
    assert payload["findings"][0]["sources"] == []
    assert not (tmp_path / "research" / "sources").exists()


# -- github issue sync -------------------------------------------------------

class _Gh:
    """Stands in for the gh CLI, recording what it was asked to do."""

    def __init__(self, exists=False):
        self.calls: list[list[str]] = []
        self.opened = 0
        self.exists = exists      # whether the remote repository is there yet

    def __call__(self, *args, cwd=None):
        self.calls.append(list(args))
        if args[:2] == ("issue", "create"):
            self.opened += 1
            return subprocess.CompletedProcess(
                args, 0, f"https://github.com/o/r/issues/{self.opened}\n", "")
        if args[:2] == ("repo", "view"):
            if not self.exists:
                return subprocess.CompletedProcess(args, 1, "", "not found")
            return subprocess.CompletedProcess(
                args, 0, '{"url": "https://github.com/o/r"}', "")
        if args[:2] == ("repo", "create"):
            self.exists = True
        return subprocess.CompletedProcess(args, 0, "", "")

    def verbs(self) -> list[str]:
        return [" ".join(call[:2]) for call in self.calls]


@pytest.fixture
def gh(monkeypatch):
    fake = _Gh()
    monkeypatch.setattr(GitHub, "_gh",
                        lambda self, *args, cwd=None: fake(*args, cwd=cwd))
    return fake


def _published(root):
    """A project that has a repository, which is what sync waits for."""
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "remote", "add", "origin",
                    "https://example.invalid/o/r.git"], cwd=root, check=True)


def _sync(root, batch=20, enabled=True):
    github = GitHub({"enabled": enabled}, NULL)
    github.available = True          # gh need not be installed to test the logic
    return Issues(github, root, NULL, batch)


def _github(**cfg):
    github = GitHub({"enabled": True, **cfg}, NULL)
    github.available = True          # gh need not be installed to test the logic
    return github


def test_creating_the_repository_publishes_no_code(tmp_path, gh):
    """The repository must exist before an issue can be opened; the code waits."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    url = _github().create(tmp_path, "proj", "a description")

    assert url == "https://github.com/o/r"
    assert "repo create" in gh.verbs()
    flat = [arg for call in gh.calls for arg in call]
    assert "--push" not in flat and "--private" in flat
    assert gh.exists


def test_an_existing_repository_is_adopted_rather_than_recreated(tmp_path):
    """Resuming a project must not try to create the repository a second time."""
    fake = _Gh(exists=True)
    original = GitHub._gh
    GitHub._gh = lambda self, *args, cwd=None: fake(*args, cwd=cwd)
    try:
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        url = _github().create(tmp_path, "proj", "d")
    finally:
        GitHub._gh = original

    assert url == "https://github.com/o/r"
    assert "repo create" not in fake.verbs()
    remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert remote.stdout.strip() == "https://github.com/o/r.git"


def test_each_ticket_becomes_an_issue(tmp_path, gh):
    workspace.create(tmp_path, "an idea")
    _published(tmp_path)
    board = Board(tmp_path)
    board.add(title="Implement calc", objective="add and divide",
              acceptance=["add works"], validation=["pytest -q"])
    board.add(title="Add tests", kind="test")
    sync = _sync(tmp_path)

    assert sync.sync(board.tickets.values()) == 2
    assert gh.verbs().count("issue create") == 2
    assert set(sync.known) == set(board.tickets)
    assert (tmp_path / ".hermes" / "state" / "github_issues.json").is_file()
    # Every issue carries the statement that it is not an input.
    assert "is not read back" in " ".join(gh.calls[-1])


def test_a_board_that_did_not_move_costs_nothing(tmp_path, gh):
    workspace.create(tmp_path, "an idea")
    _published(tmp_path)
    board = Board(tmp_path)
    board.add(title="Implement calc")
    sync = _sync(tmp_path)
    sync.sync(board.tickets.values())
    spent = len(gh.calls)

    assert sync.sync(board.tickets.values()) == 0
    assert len(gh.calls) == spent


def test_a_finished_ticket_closes_its_issue(tmp_path, gh):
    workspace.create(tmp_path, "an idea")
    _published(tmp_path)
    board = Board(tmp_path)
    ticket = board.add(title="Implement calc")
    sync = _sync(tmp_path)
    sync.sync(board.tickets.values())

    board.move(ticket.id, "completed", "gates passed")
    sync.sync(board.tickets.values())

    assert "issue close" in gh.verbs()
    assert sync.known[ticket.id]["closed"] is True


def test_the_projection_never_reads_an_issue_back(tmp_path, gh):
    """Two-way sync would put a human inside a loop meant to run unattended."""
    workspace.create(tmp_path, "an idea")
    _published(tmp_path)
    board = Board(tmp_path)
    ticket = board.add(title="Implement calc")
    sync = _sync(tmp_path)
    sync.sync(board.tickets.values())
    board.move(ticket.id, "active", "picked up")
    sync.sync(board.tickets.values())

    assert not [v for v in gh.verbs() if v in ("issue view", "issue list")]


def test_a_project_without_a_repository_makes_no_calls(tmp_path, gh):
    workspace.create(tmp_path, "an idea")        # no git remote yet
    board = Board(tmp_path)
    board.add(title="Implement calc")

    assert _sync(tmp_path).sync(board.tickets.values()) == 0
    assert not [v for v in gh.verbs() if v.startswith("issue")]


def test_disabled_github_reaches_nothing(tmp_path, gh):
    workspace.create(tmp_path, "an idea")
    _published(tmp_path)
    board = Board(tmp_path)
    board.add(title="Implement calc")

    assert _sync(tmp_path, enabled=False).sync(board.tickets.values()) == 0
    assert gh.calls == []


def test_a_freshly_planned_board_spreads_over_iterations(tmp_path, gh):
    workspace.create(tmp_path, "an idea")
    _published(tmp_path)
    board = Board(tmp_path)
    for n in range(5):
        board.add(title=f"ticket {n}")
    sync = _sync(tmp_path, batch=2)

    assert sync.sync(board.tickets.values()) == 2
    assert sync.sync(board.tickets.values()) == 2
    assert sync.sync(board.tickets.values()) == 1
    assert gh.verbs().count("issue create") == 5


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


def _engine(tmp_path, cfg=None, direction="minimize"):
    state = SimpleNamespace(plan={}, spec={}, name="p", idea="i")
    return Evolution(Sandbox(tmp_path, {"runtime": "local", "timeout": 5}), {}, state,
                     Objective("time", direction, "true"), ["src/a.py"], cfg or {})


def test_measurement_noise_is_not_an_improvement(tmp_path):
    """One run is a measurement; promoting its wobble stalls the whole search."""
    engine = _engine(tmp_path, {"min_relative_gain": 0.0})
    engine.noise = 0.5
    assert not engine.significant(9.7, 10.0)
    assert engine.significant(9.4, 10.0)


def test_a_gain_too_small_to_be_worth_the_risk_is_rejected(tmp_path):
    engine = _engine(tmp_path, {"min_relative_gain": 0.05})
    assert not engine.significant(9.9, 10.0)
    assert engine.significant(9.4, 10.0)


def test_archive_ranks_by_direction_and_is_capped(tmp_path):
    engine = _engine(tmp_path, {"archive": 2}, direction="maximize")
    for index, score in enumerate([1.0, 5.0, 3.0], start=1):
        engine.history.append(Candidate(index, 0, "m", score, True, []))
    assert [c.score for c in engine.archive()][:2] == [5.0, 3.0]


def test_screening_bounds_a_run_by_the_incumbent(tmp_path):
    engine = _engine(tmp_path, {"benchmark_timeout": 900, "screen_factor": 2.0})
    assert engine.screen_timeout() == 900  # nothing measured yet
    engine.best_seconds = 100.0
    assert engine.screen_timeout() == 205
    engine.best_seconds = 10000.0
    assert engine.screen_timeout() == 900  # never beyond the hard budget


def test_evolution_promotes_only_a_gain_that_clears_the_margin(tmp_path):
    """The whole loop: propose, measure, reject noise, confirm, promote."""
    sandbox = Sandbox(tmp_path, {"runtime": "local", "timeout": 30})
    sandbox.write("src/kernel.py", "COST = 10\n")
    state = SimpleNamespace(plan={"commands": {"build": "true", "test": "true"}},
                            spec={}, name="p", idea="i")
    objective = Objective("cost", "minimize",
                          "python3 -c \"exec(open('src/kernel.py').read()); "
                          "print('SCORE:', COST)\"")
    proposals = [
        {"mutation": "shave a little",
         "files": [{"path": "src/kernel.py", "content": "COST = 9.95\n"}]},
        {"mutation": "halve the cost",
         "files": [{"path": "src/kernel.py", "content": "COST = 5\n"}]},
    ]

    class StubEvolver:
        def ask_json(self, prompt, **kw):
            return proposals.pop(0)

    engine = Evolution(sandbox, {"evolver": StubEvolver()}, state, objective,
                       ["src/kernel.py"],
                       {"population": 1, "baseline_repeats": 2,
                        "min_relative_gain": 0.01}, NULL)
    best = engine.run(max_generations=2, patience=5)

    assert [c.mutation for c in engine.history] == [
        "baseline", "shave a little", "halve the cost"]
    # 0.05 is under the 1% margin, so it is not an improvement, it is weather.
    assert engine.history[1].notes.startswith("no significant gain")
    assert best.score == 5 and sandbox.read("src/kernel.py") == "COST = 5\n"
    # The gain was measured more than once before it was believed.
    assert len(best.samples) == 2


def test_bandit_tries_every_family_once_before_repeating():
    bandit = Bandit(("a", "b", "c"))
    for _ in range(3):
        bandit.update(bandit.select(), 0.0)
    assert set(bandit.counts.values()) == {1}


def test_bandit_concentrates_on_the_family_that_paid_off():
    bandit = Bandit(("a", "b"))
    for _ in range(3):
        bandit.update("a", 0.0)
        bandit.update("b", 1.0)
    assert bandit.select() == "b"


# -- evaluation harness ------------------------------------------------------

def test_eval_counts_first_attempt_completions(tmp_path):
    board = Board(tmp_path)
    for attempts in (1, 1, 3):
        board.add(title="t", status="completed").attempts = attempts
    board.add(title="dead end", status="abandoned")
    trial = harness.collect("demo", SimpleNamespace(phase="COMPLETED"), board,
                            [{"seconds": 12.0, "chars_out": 100}], 30.0, 100.0)
    assert trial.completed and trial.tickets_completed == 3
    assert trial.tickets_abandoned == 1
    assert trial.first_pass == pytest.approx(2 / 3)
    assert trial.inference_seconds == 12.0 and trial.sandbox_seconds == 30.0
    assert trial.model_calls == 1


def test_eval_writes_both_report_formats(tmp_path):
    trial = harness.Trial(name="demo", completed=True, seconds=60.0,
                          tickets=2, tickets_completed=2, first_pass=1.0)
    report = harness.write_report(tmp_path, [trial])
    assert report.is_file() and "demo" in report.read_text()
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["trials"][0]["name"] == "demo"
    assert "completed      1/1" in harness.summarise([trial])


def test_eval_reports_the_clock_neither_ledger_explains(tmp_path):
    """The GPU/CPU split is only trustworthy if the remainder is shown."""
    trial = harness.collect("demo", SimpleNamespace(phase="COMPLETED", iteration=7),
                            Board(tmp_path), [{"seconds": 12.0}], 30.0, 100.0)
    assert trial.unaccounted_seconds == 58.0
    assert trial.overlap_seconds == 0.0
    assert trial.iterations == 7
    assert "unaccounted    1.0m" in harness.summarise([trial])


def test_eval_reports_ledgers_that_overlap_rather_than_hiding_them(tmp_path):
    """Prefetch generates while gates run, so the ledgers may exceed the clock."""
    trial = harness.collect("demo", SimpleNamespace(phase="COMPLETED"),
                            Board(tmp_path), [{"seconds": 80.0}], 60.0, 100.0)
    assert trial.overlap_seconds == 40.0
    assert trial.unaccounted_seconds == 0.0


def test_eval_collects_a_run_that_never_got_going(tmp_path):
    """A crashed or timed-out entry is a result: reading it must not raise."""
    trial = harness.collect("demo", SimpleNamespace(), Board(tmp_path), [], 0.0,
                            5.0, error="boom", timed_out=True)
    assert trial.phase == "UNKNOWN" and not trial.completed
    assert trial.iterations == 0 and trial.timed_out
    assert "TIME" in trial.row()
    assert "timed out      1 task(s)" in harness.summarise([trial])


def test_eval_says_when_a_suite_cannot_rank_anything():
    """A suite where everything passes identically cannot compare two factories."""
    same = [harness.Trial(name=n, completed=True, tickets_completed=2, first_pass=1.0)
            for n in ("a", "b")]
    assert "resolution     NONE" in harness.summarise(same)
    assert "cannot tell two factories apart" in harness.summarise(same)

    none_done = [harness.Trial(name=n) for n in ("a", "b")]
    assert "no task completed" in harness.summarise(none_done)

    mixed = [harness.Trial(name="a", completed=True, tickets_completed=2,
                           first_pass=1.0),
             harness.Trial(name="b", completed=True, tickets_completed=2,
                           first_pass=0.5)]
    assert "resolution     2/2 completed, 2 distinct" in harness.summarise(mixed)


def test_eval_warns_that_a_stub_run_measures_nothing():
    """Numbers produced without a model describe the stub, not the factory."""
    trials = [harness.Trial(name="a", completed=True, tickets_completed=1,
                            first_pass=1.0),
              harness.Trial(name="b", model_calls=0)]
    assert "no model calls recorded" in harness.summarise(trials)

    with_model = [harness.Trial(name="a", completed=True, model_calls=3)]
    assert "no model calls recorded" not in harness.summarise(with_model)


def test_eval_compare_refuses_to_read_meaning_into_an_unmoved_pair(tmp_path):
    """Two runs that agree everywhere cannot justify accepting a change."""
    def report(name, **over):
        directory = tmp_path / name
        trial = harness.Trial(name="json-diff", completed=True, seconds=60.0,
                              tickets_completed=2, first_pass=1.0, model_calls=9)
        for key, value in over.items():
            setattr(trial, key, value)
        harness.write_report(directory, [trial])
        return directory / "results.json"

    unmoved = harness.compare(report("before"), report("after"))
    assert "does not support accepting or rejecting" in unmoved

    moved = harness.compare(report("b2"), report("a2", first_pass=0.5))
    assert "1 of 1 task(s) moved: json-diff" in moved

    stubs = harness.compare(report("b3", model_calls=0), report("a3", model_calls=0))
    assert "compares two stubs" in stubs


# -- self-improvement --------------------------------------------------------

def _repo(root):
    """A git repository shaped like Hermes, small enough to branch in a test."""
    (root / "hermes").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "hermes" / "planning.py").write_text("VERSION = 1\n")
    (root / "tests" / "test_units.py").write_text("def test_ok():\n    assert True\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.invalid"], cwd=root,
                   check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=root, check=True)


def test_a_change_to_what_judges_it_is_refused_unevaluated():
    """The cheapest way to pass a suite is to delete it, so this is refused."""
    from hermes.improve import admissible

    assert "judges it" in admissible(["tests/test_units.py"])
    assert "judges it" in admissible(["hermes/harness.py"])
    assert "judges it" in admissible(["hermes/improve.py"])
    assert "judges it" in admissible(["pyproject.toml"])
    assert admissible(["hermes/planning.py"]) == ""
    assert "no files" in admissible([])
    # Hermes improves Hermes, not the README or the installer.
    assert "outside the hermes package" in admissible(["README.md"])


def test_an_unmeasured_change_is_never_promotable(tmp_path):
    """Section 35 permits promotion on demonstrated benefit, nothing less."""
    from hermes.improve import Improvement

    gate = Improvement(tmp_path).benefit(None, None)
    assert not gate.passed
    assert "demonstrated" in gate.detail


def test_measured_gain_reads_two_evaluations_strictly(tmp_path):
    from hermes import harness
    from hermes.improve import measured_gain

    def report(name, **over):
        trial = harness.Trial(name="json-diff", completed=True, seconds=60.0,
                              tickets_completed=2, first_pass=0.5, model_calls=9)
        for key, value in over.items():
            setattr(trial, key, value)
        harness.write_report(tmp_path / name, [trial])
        return tmp_path / name / "results.json"

    base = report("base")
    assert measured_gain(base, report("better", first_pass=1.0))[0]
    assert not measured_gain(base, report("same"))[0]
    assert not measured_gain(base, report("worse", completed=False))[0]
    # Two stub runs describe the stub, not the factory.
    stub_a, stub_b = report("sa", model_calls=0), report("sb", model_calls=0)
    ok, why = measured_gain(stub_a, stub_b)
    assert not ok and "stubs" in why


def test_a_candidate_is_built_beside_the_live_tree_not_in_it(tmp_path):
    """Hermes runs from its checkout; a candidate must not rewrite it."""
    from hermes.improve import Improvement

    root = tmp_path / "repo"
    root.mkdir()
    _repo(root)
    work = Improvement(root)

    worktree = work.worktree("hermes/improve-planner-1")
    written = work.apply(worktree, [{"path": "hermes/planning.py",
                                     "content": "VERSION = 2\n"}])

    assert written == ["hermes/planning.py"]
    assert (worktree / "hermes" / "planning.py").read_text() == "VERSION = 2\n"
    # The tree Hermes is executing from is untouched.
    assert (root / "hermes" / "planning.py").read_text() == "VERSION = 1\n"

    work.discard("hermes/improve-planner-1")
    assert not worktree.exists()


def test_the_suite_that_judges_a_candidate_is_always_ours(tmp_path):
    """A candidate that reached the tests by any route has them taken back."""
    from hermes.improve import Improvement

    root = tmp_path / "repo"
    root.mkdir()
    _repo(root)
    work = Improvement(root)
    worktree = work.worktree("hermes/improve-planner-2")

    (worktree / "tests" / "test_units.py").write_text("def test_ok():\n    pass\n")
    work.restore_trusted(worktree)

    assert (worktree / "tests" / "test_units.py").read_text() == \
        (root / "tests" / "test_units.py").read_text()
    work.discard("hermes/improve-planner-2")


def test_an_escape_from_the_worktree_is_refused(tmp_path):
    from hermes.improve import Improvement

    root = tmp_path / "repo"
    root.mkdir()
    _repo(root)
    work = Improvement(root)
    worktree = work.worktree("hermes/improve-planner-3")

    written = work.apply(worktree, [
        {"path": "../escaped.py", "content": "nope"},
        {"path": "/etc/escaped.py", "content": "nope"},
        {"path": "hermes/planning.py", "content": "VERSION = 3\n"}])

    assert written == ["hermes/planning.py"]
    assert not (tmp_path / "escaped.py").exists()
    work.discard("hermes/improve-planner-3")


def test_nothing_is_promoted_without_passing_every_gate(tmp_path):
    from hermes.improve import Attempt, Improvement
    from hermes.verify import Gate

    root = tmp_path / "repo"
    root.mkdir()
    _repo(root)
    work = Improvement(root)

    failed = Attempt(target="planner", branch="hermes/improve-planner-4",
                     gates=[Gate("suite", True), Gate("benefit", False, "unmeasured")])
    assert not work.promote(failed)
    assert not failed.promoted

    never_evaluated = Attempt(target="planner")
    assert not never_evaluated.passed
    assert not work.promote(never_evaluated)


def test_an_unknown_target_is_rejected(tmp_path):
    from hermes.improve import Improvement

    record = Improvement(tmp_path).attempt("the whole thing")
    assert "unknown target" in record.refusal
    assert not record.passed


# -- released tree -----------------------------------------------------------

def _with_retrieved_pages(root):
    """A project holding both retrieved evidence and a directory merely named
    `sources`, which is far too ordinary a name to exclude wherever it lands."""
    workspace.create(root, "an idea")
    (root / "research" / "sources").mkdir(parents=True, exist_ok=True)
    (root / "research" / "sources" / "guide.html").write_text("<p>someone else's</p>")
    (root / "research" / "notes.md").write_text("cited <http://example.invalid/g>")
    (root / "src" / "sources").mkdir(parents=True, exist_ok=True)
    (root / "src" / "sources" / "feed.py").write_text("FEED = 1\n")


def test_release_never_ships_retrieved_research_pages(tmp_path):
    """Evidence is for re-reading, not for republishing under our own licence."""
    from hermes import release
    root = tmp_path / "proj"
    root.mkdir()
    _with_retrieved_pages(root)
    # Committed on purpose: a project created before the pages were gitignored
    # must still not ship them.
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A", "-f"], cwd=root, check=True)

    out = tmp_path / "export"
    release.export(root, out)
    assert not (out / "research" / "sources").exists()
    assert (out / "research" / "notes.md").is_file()      # the citation ships
    assert (out / "src" / "sources" / "feed.py").is_file()  # the name is not the rule


def test_release_drops_retrieved_pages_from_an_untracked_tree_too(tmp_path):
    """The copytree fallback runs when there is no git; it must exclude them too."""
    from hermes import release
    root = tmp_path / "proj"
    root.mkdir()
    _with_retrieved_pages(root)

    out = tmp_path / "export"
    release.export(root, out)
    assert not (out / "research" / "sources").exists()
    assert (out / "src" / "sources" / "feed.py").is_file()
    assert release.shipped("research/notes.md")
    assert not release.shipped("research/sources/guide.html")


def test_generated_projects_never_commit_retrieved_pages(tmp_path):
    """Gitignored, so the pages are not pushed to GitHub either."""
    workspace.create(tmp_path, "an idea")
    assert "research/sources/" in (tmp_path / ".gitignore").read_text()

    (tmp_path / "research" / "sources").mkdir(parents=True, exist_ok=True)
    (tmp_path / "research" / "sources" / "guide.html").write_text("<p>theirs</p>")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    listed = subprocess.run(["git", "ls-files"], cwd=tmp_path,
                            capture_output=True, text=True).stdout
    assert "research/sources" not in listed
    assert ".gitignore" in listed


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


def test_console_script_triggers_editable_install(shared_tmp):
    """A missing console script means the project is not installed, not broken."""
    import shutil
    if not shutil.which("docker"):
        pytest.skip("docker unavailable")
    (shared_tmp / "greet.py").write_text("def main():\n    print('hi')\n")
    (shared_tmp / "setup.py").write_text(
        "from setuptools import setup\n"
        "setup(name='greet', version='0', py_modules=['greet'],\n"
        "      entry_points={'console_scripts': ['greet=greet:main']})\n")
    plan = {"commands": {"build": "true", "test": "greet | grep -q hi"}}
    sandbox = Sandbox(shared_tmp, {"runtime": "docker", "timeout": 300,
                                   "image": "python:3.11-slim"})
    with sandbox:
        if sandbox.runtime != "docker":
            pytest.skip("docker present but unusable")
        verification = verify_project(sandbox, plan)
    assert verification.passed, verification.report()
    assert plan["commands"]["setup"] == "pip install -e ."


def test_unknown_tool_is_not_fetched_from_pypi(shared_tmp):
    """Model-written command text must not become a PyPI install."""
    import shutil
    if not shutil.which("docker"):
        pytest.skip("docker unavailable")
    plan = {"commands": {"build": "true", "test": "definitely-not-a-tool --run"}}
    sandbox = Sandbox(shared_tmp, {"runtime": "docker", "timeout": 120,
                                   "image": "python:3.11-slim"})
    with sandbox:
        if sandbox.runtime != "docker":
            pytest.skip("docker present but unusable")
        verification = verify_project(sandbox, plan)
    assert not verification.passed
    assert sandbox.installed == []


def test_failed_install_reaches_the_agent(shared_tmp):
    """The install error is the diagnosis, so it must survive into the report."""
    import shutil
    if not shutil.which("docker"):
        pytest.skip("docker unavailable")
    (shared_tmp / "src").mkdir()
    (shared_tmp / "src" / "greet.py").write_text("def main():\n    print('hi')\n")
    # packages=['greet'] with the code under src/ and no package_dir: the
    # editable install cannot work, and saying so is the point.
    (shared_tmp / "setup.py").write_text(
        "from setuptools import setup\n"
        "setup(name='greet', version='0', packages=['greet'],\n"
        "      entry_points={'console_scripts': ['greet=greet:main']})\n")
    plan = {"commands": {"build": "true", "test": "greet"}}
    sandbox = Sandbox(shared_tmp, {"runtime": "docker", "timeout": 300,
                                   "image": "python:3.11-slim"})
    with sandbox:
        if sandbox.runtime != "docker":
            pytest.skip("docker present but unusable")
        verification = verify_project(sandbox, plan)
    assert not verification.passed
    assert "pip install" in verification.failure_output()


def test_prose_validation_becomes_an_acceptance_criterion():
    """A planner sentence in the validation list can only ever fail as a command."""
    commands, prose = split_validation([
        "pytest tests -q",
        "All standard subtractive pairs (IV, IX, XL, XC, CD, CM) pass",
        "All test cases pass without false positives",
        "python -c 'import roman'",
        "make build",
        "cargo test --all-features",
        "bash scripts/check.sh one two three four",
    ])
    assert commands == ["pytest tests -q", "python -c 'import roman'", "make build",
                        "cargo test --all-features",
                        "bash scripts/check.sh one two three four"]
    assert len(prose) == 2


# -- hpf entry point ---------------------------------------------------------

@pytest.fixture
def captured_main(monkeypatch):
    """Record the argv hpf hands to the main CLI instead of running a factory."""
    seen: dict = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(cli, "main", fake_main)
    return seen


def test_hpf_starts_a_project_in_the_current_directory(tmp_path, monkeypatch,
                                                       captured_main):
    monkeypatch.chdir(tmp_path)
    assert cli.hpf(["Build", "a", "trading", "engine"]) == 0
    assert captured_main["argv"] == ["new", "Build a trading engine",
                                     "--into", str(tmp_path)]


def test_hpf_without_an_idea_resumes_the_project_here(tmp_path, monkeypatch,
                                                      captured_main):
    workspace.create(tmp_path, "an idea")
    monkeypatch.chdir(tmp_path)
    assert cli.hpf([]) == 0
    assert captured_main["argv"] == ["--project", str(tmp_path), "run"]


def test_hpf_without_an_idea_or_a_project_explains_itself(tmp_path, monkeypatch,
                                                          captured_main):
    monkeypatch.chdir(tmp_path)
    assert cli.hpf([]) == 2
    assert "argv" not in captured_main


def test_hpf_refuses_to_overwrite_an_existing_project(tmp_path, monkeypatch,
                                                      captured_main):
    workspace.create(tmp_path, "first idea")
    monkeypatch.chdir(tmp_path)
    assert cli.hpf(["a", "second", "idea"]) == 1
    assert "argv" not in captured_main
    assert cli.hpf(["a", "second", "idea", "--force"]) == 0
    assert "--force" in captured_main["argv"]


def test_hpf_places_global_and_subcommand_flags_correctly(tmp_path, monkeypatch,
                                                          captured_main):
    """-q is a global flag, -n belongs to the subcommand; argparse is strict."""
    monkeypatch.chdir(tmp_path)
    cli.hpf(["an idea", "-q", "-n", "5"])
    assert captured_main["argv"] == ["-q", "new", "an idea", "--into",
                                     str(tmp_path), "-n", "5"]


# -- model server ------------------------------------------------------------

class _Systemctl:
    """Stands in for `systemctl --user`, recording what was asked of it."""

    def __init__(self, *, unit_exists=True, start_ok=True):
        self.calls: list[list[str]] = []
        self.unit_exists = unit_exists
        self.start_ok = start_ok
        self.active = False

    def __call__(self, args):
        self.calls.append(args)
        verb = args[0]
        if verb == "cat":
            return (0, "[Unit]\n") if self.unit_exists else (1, "No files found")
        if verb == "start":
            if not self.start_ok:
                return 1, "Job for hermes-vllm.service failed"
            self.active = True
            return 0, ""
        if verb == "stop":
            self.active = False
            return 0, ""
        if verb == "is-active":
            return (0, "active\n") if self.active else (3, "inactive\n")
        return 0, ""

    @property
    def verbs(self):
        return [call[0] for call in self.calls]


def _router_on(base_url, up):
    """A Router stand-in: two roles behind one endpoint, up when *up* says so."""
    def available_models():
        if not up():
            raise ModelError(f"{base_url}: connection refused")
        return ["qwen2.5-coder-7b-instruct-awq"]
    backend = SimpleNamespace(cfg=SimpleNamespace(base_url=base_url),
                              available_models=available_models)
    return SimpleNamespace(routed_tasks=lambda: ["coding", "review"],
                           backend=lambda task: backend)


def _server(sysctl, **kw):
    from hermes.server import ModelServer
    kw.setdefault("wait", 60)
    return ModelServer("hermes-vllm.service", log=NULL, run=sysctl,
                       sleep=lambda _s: None, **kw)


def test_the_server_runs_for_the_length_of_a_run():
    """Started when the project starts, stopped when it ends.

    vLLM pins the whole model in VRAM, so "up all the time" costs gigabytes of
    GPU while nothing is building. The unit install.sh writes is therefore not
    enabled, and this is what brings it up instead.
    """
    sysctl = _Systemctl()
    server = _server(sysctl)
    server.ensure(_router_on("http://localhost:6767/v1", lambda: sysctl.active))
    assert server.started and sysctl.active
    assert sysctl.verbs == ["cat", "start"]
    server.release()
    assert not sysctl.active
    assert sysctl.verbs == ["cat", "start", "stop"]


def test_a_server_hermes_did_not_start_is_left_running():
    """Someone who started the server by hand did it for a reason."""
    sysctl = _Systemctl()
    sysctl.active = True
    server = _server(sysctl)
    server.ensure(_router_on("http://localhost:6767/v1", lambda: sysctl.active))
    assert sysctl.calls == [] and not server.started
    server.release()
    assert sysctl.calls == [] and sysctl.active


def test_a_remote_endpoint_is_not_ours_to_start():
    sysctl = _Systemctl()
    server = _server(sysctl)
    server.ensure(_router_on("http://gpu.example.com/v1", lambda: False))
    assert sysctl.calls == [] and not server.started


def test_a_missing_unit_is_not_an_error_here():
    """preflight reports the unreachable server; this stays quiet."""
    sysctl = _Systemctl(unit_exists=False)
    server = _server(sysctl)
    server.ensure(_router_on("http://localhost:6767/v1", lambda: False))
    assert sysctl.verbs == ["cat"] and not server.started
    server.release()
    assert sysctl.verbs == ["cat"]


def test_a_failed_start_is_not_followed_by_a_stop():
    sysctl = _Systemctl(start_ok=False)
    server = _server(sysctl)
    server.ensure(_router_on("http://localhost:6767/v1", lambda: sysctl.active))
    assert sysctl.verbs == ["cat", "start"] and not server.started
    server.release()
    assert sysctl.verbs == ["cat", "start"]


def test_a_server_that_never_answers_is_still_released():
    """The weights may never load; the process is ours either way."""
    sysctl = _Systemctl()
    server = _server(sysctl, wait=0)
    server.ensure(_router_on("http://localhost:6767/v1", lambda: False))
    assert server.started
    server.release()
    assert sysctl.verbs == ["cat", "start", "stop"] and not sysctl.active


def test_stop_after_run_false_keeps_the_model_resident():
    sysctl = _Systemctl()
    server = _server(sysctl, stop_after_run=False)
    server.ensure(_router_on("http://localhost:6767/v1", lambda: sysctl.active))
    server.release()
    assert sysctl.verbs == ["cat", "start"] and sysctl.active


def test_autostart_off_means_no_server_object_at_all():
    from hermes.config import DEFAULTS
    from hermes.server import ModelServer
    assert DEFAULTS["server"]["autostart"] is True
    assert ModelServer.from_config(Config(deep_merge(DEFAULTS, {}))).unit \
        == "hermes-vllm.service"
    off = Config(deep_merge(DEFAULTS, {"server": {"autostart": False}}))
    assert ModelServer.from_config(off) is None
    unitless = Config(deep_merge(DEFAULTS, {"server": {"unit": ""}}))
    assert ModelServer.from_config(unitless) is None


def test_unit_state_calls_an_idle_unit_inactive_not_missing():
    """`hermes doctor` must tell "installed, down between runs" from "absent"."""
    from hermes.server import unit_state
    sysctl = _Systemctl()
    assert unit_state("hermes-vllm.service", run=sysctl) == "inactive"
    sysctl.active = True
    assert unit_state("hermes-vllm.service", run=sysctl) == "active"
    assert unit_state("x.service", run=_Systemctl(unit_exists=False)) == "not installed"


def test_doctor_excuses_an_idle_server_but_never_a_bad_config(tmp_path, monkeypatch, capsys):
    """An installed unit that is down between runs is the normal resting state.

    Port 1 stands in for "nothing listening": the point is that doctor decides
    on the config first, so the local-only rule and an empty model name still
    fail while the server is stopped.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hermes.server.unit_state", lambda unit, run=None: "inactive")
    config = tmp_path / "hermes.yaml"
    base = ("models:\n  default:\n    backend: vllm\n"
            "    base_url: {url}\n    model: {model}\n"
            "server:\n  autostart: {autostart}\n  unit: hermes-vllm.service\n")
    args = SimpleNamespace(config=None)

    config.write_text(base.format(url="http://127.0.0.1:1/v1", model="served",
                                  autostart="true"))
    assert cli.cmd_doctor(args) == 0
    assert "the next run starts it" in capsys.readouterr().out

    # Off: nothing will start it for you, so an unreachable server is a fault.
    config.write_text(base.format(url="http://127.0.0.1:1/v1", model="served",
                                  autostart="false"))
    assert cli.cmd_doctor(args) == 1

    config.write_text(base.format(url="http://127.0.0.1:1/v1", model='""',
                                  autostart="true"))
    assert cli.cmd_doctor(args) == 1

    config.write_text(base.format(url="https://api.openai.com/v1", model="gpt-4",
                                  autostart="true"))
    assert cli.cmd_doctor(args) == 1
    assert "not a local endpoint" in capsys.readouterr().err


def test_running_holds_the_server_for_exactly_one_block(monkeypatch):
    """The contextmanager every generating entry point wraps its work in."""
    from hermes.config import DEFAULTS
    from hermes.server import running
    sysctl = _Systemctl()
    monkeypatch.setattr("hermes.server._systemctl", sysctl)
    router = _router_on("http://localhost:6767/v1", lambda: sysctl.active)
    with running(Config(deep_merge(DEFAULTS, {})), router) as server:
        assert sysctl.active and server.started
    assert not sysctl.active
    assert sysctl.verbs == ["cat", "start", "stop"]

    off = Config(deep_merge(DEFAULTS, {"server": {"autostart": False}}))
    with running(off, router) as server:
        assert server is None
