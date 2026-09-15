"""Unit checks for the logic that has to be right for the loop to terminate."""
import json
import threading
import time

import pytest

from types import SimpleNamespace

from hermes.config import Config, ConfigError, deep_merge, is_local_url
from hermes.evolve import Bandit, Candidate, Evolution, Objective, parse_score
from hermes.implement import (Change, _protected, _race, apply_change, execute_ticket,
                              relevant_files, restore as restore_files,
                              snapshot as snapshot_files)
from hermes.journal import Entry, Journal
from hermes.log import NULL
from hermes.models import ModelError, Router, extract_json, strip_reasoning
from hermes.openhands import OpenHands
from hermes.resources import Governor
from hermes import harness
from hermes.sandbox import Sandbox, SandboxError
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
