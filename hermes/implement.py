"""Ticket execution: generate code, apply it, verify, repair, review."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import prompts
from .agents import Prompt, context_block, escalate, file_protocol, ticket_block
from .anchor import locate
from .ledger import ABANDONED, FAILED, FIXED, NullLedger
from .log import Logger, NULL, scope
from .sandbox import Sandbox, SandboxError
from .tools import Lookup, Toolbox
from .verify import Verification, review, triage, verify_ticket

#: Files an agent may never write, regardless of what it asks for.
PROTECTED = (".hermes/", ".git/", "hermes.yaml")


@dataclass(frozen=True)
class Rung:
    """How hard one attempt at a ticket tries.

    *candidates* is how many samples are raced through the gates, *files* how
    many existing sources the prompt may show, and *role* which agent is
    asked -- empty meaning the ticket's own implementer or tester.
    """

    candidates: int = 1
    files: int = 6
    role: str = ""

    @property
    def spec(self) -> tuple[int, int, str]:
        """This rung as the ledger stores it. See :mod:`hermes.ledger`."""
        return (self.candidates, self.files, self.role)

    def harder(self, spec: dict) -> "Rung":
        """This rung with whatever *spec* names overridden, nothing else."""
        return Rung(candidates=max(1, int(spec.get("candidates", self.candidates))),
                    files=max(1, int(spec.get("files", self.files))),
                    role=str(spec.get("role", self.role)))


@dataclass
class Ladder:
    """What each successive attempt at a ticket is allowed to spend.

    A failure that is retried identically is a retry, not an escalation: the
    same model, the same prompt and the same decoding produce the same answer
    often enough that three attempts cost three times one attempt's chance.
    So attempt two races more samples and is shown more of the code, and
    attempt three is asked of a reasoning role rather than the implementer.

    Later rungs inherit the one below them, so a ladder never quietly
    de-escalates, and the top rung repeats once the list runs out: effort
    stops growing rather than dropping back to the cheapest ask.
    """

    rungs: list[Rung] = field(default_factory=lambda: [Rung()])

    def rung(self, attempt: int) -> Rung:
        return self.rungs[min(max(attempt, 1), len(self.rungs)) - 1]

    @property
    def escalates(self) -> bool:
        return len(self.rungs) > 1

    @classmethod
    def from_config(cls, config) -> "Ladder":
        """Build the ladder from ``execution.candidates/files/escalation``."""
        rungs = [Rung(candidates=max(1, int(config.get("execution.candidates", 1))),
                      files=max(1, int(config.get("execution.files", 6))))]
        for spec in config.get("execution.escalation") or []:
            if isinstance(spec, dict):
                rungs.append(rungs[-1].harder(spec))
        return cls(rungs)


@dataclass
class Change:
    written: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    summary: str = ""
    dependencies: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    #: Edits whose "find" text did not match, phrased for the next attempt.
    unmatched: list[str] = field(default_factory=list)
    #: Edits that only landed because the applier widened the search, with how.
    #: Nothing depends on this; it is what says whether widening pays.
    repaired: list[str] = field(default_factory=list)

    @property
    def touched(self) -> list[str]:
        return self.written + self.deleted

    @property
    def empty(self) -> bool:
        return not self.written and not self.deleted

    def describe(self) -> str:
        parts = []
        if self.written:
            parts.append(f"wrote {', '.join(self.written[:8])}")
        if self.deleted:
            parts.append(f"deleted {', '.join(self.deleted[:5])}")
        if self.rejected:
            parts.append(f"rejected {', '.join(self.rejected[:5])}")
        if self.repaired:
            parts.append(f"re-anchored {', '.join(self.repaired[:5])}")
        if self.unmatched:
            parts.append(f"{len(self.unmatched)} edit(s) did not match")
        return "; ".join(parts) or "no changes"


def apply_change(sandbox: Sandbox, payload: dict, log: Logger = NULL) -> Change:
    """Write the files an agent asked for, refusing anything outside the project."""
    change = Change(summary=str(payload.get("summary", ""))[:500],
                    dependencies=[str(d) for d in (payload.get("dependencies") or [])],
                    commands=[str(c) for c in (payload.get("commands") or [])])
    for item in payload.get("files") or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip().lstrip("/")
        content = item.get("content")
        if not path or content is None:
            continue
        if _protected(path):
            change.rejected.append(path)
            log.warn("implement.rejected", path=path, reason="protected path")
            continue
        try:
            sandbox.write(path, content if isinstance(content, str)
                          else json.dumps(content, indent=2))
        except SandboxError as exc:
            change.rejected.append(path)
            log.warn("implement.rejected", path=path, reason=str(exc))
            continue
        change.written.append(path)
    for item in payload.get("edits") or []:
        if isinstance(item, dict):
            _apply_edit(sandbox, item, change, log)
    for path in payload.get("delete") or []:
        path = str(path).strip().lstrip("/")
        if not path or _protected(path):
            change.rejected.append(path)
            continue
        try:
            target = sandbox.resolve(path)
        except SandboxError:
            change.rejected.append(path)
            continue
        if target.is_file():
            target.unlink()
            change.deleted.append(path)
    log.info("implement.applied", files=len(change.written), deleted=len(change.deleted),
             rejected=len(change.rejected), unmatched=len(change.unmatched),
             repaired=len(change.repaired))
    return change


def _apply_edit(sandbox: Sandbox, item: dict, change: Change, log: Logger) -> None:
    """Apply one search/replace edit, or record precisely why it did not apply.

    An edit is only safe when its anchor is unambiguous, so a "find" that is
    absent, or that could mean two places, is refused rather than guessed at.
    Saying which of those happened is what lets the next attempt fix it instead
    of resending the same edit. :mod:`hermes.anchor` decides where -- and
    whether -- the anchor lands; everything here is about the file.
    """
    path = str(item.get("path", "")).strip().lstrip("/")
    find, replace = item.get("find"), item.get("replace")
    if not path or not isinstance(find, str) or not isinstance(replace, str) or not find:
        return
    if _protected(path):
        change.rejected.append(path)
        log.warn("implement.rejected", path=path, reason="protected path")
        return
    try:
        text = sandbox.read(path)
    except SandboxError as exc:
        change.rejected.append(path)
        log.warn("implement.rejected", path=path, reason=str(exc))
        return
    if not text:
        change.unmatched.append(
            f"{path}: the file does not exist yet, so it cannot be edited. "
            "Return its complete contents under \"files\".")
        return
    found, reason = locate(text, find, replace, line=_int(item.get("line")),
                           symbol=str(item.get("symbol") or ""))
    if found is None:
        change.unmatched.append(f"{path}: {reason}")
        return
    sandbox.write(path, found.splice(text))
    if not found.exact:
        change.repaired.append(f"{path} ({found.how})")
        log.info("implement.edit_relocated", path=path, how=found.how)
    if path not in change.written:
        change.written.append(path)


def _int(value) -> int | None:
    """*value* as a line number, for a hint a model may send as a string."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _protected(path: str) -> bool:
    normalised = path.replace("\\", "/")
    return (normalised.startswith(PROTECTED) or normalised.startswith("../")
            or "/../" in normalised or Path(normalised).is_absolute())


def targets_of(payload: dict) -> list[str]:
    """Every path a payload intends to touch, before any of it is applied."""
    paths: list[str] = []
    for key in ("files", "edits"):
        for item in payload.get(key) or []:
            if isinstance(item, dict) and str(item.get("path", "")).strip():
                paths.append(str(item["path"]).strip().lstrip("/"))
    paths += [str(p).strip().lstrip("/") for p in payload.get("delete") or [] if str(p).strip()]
    return list(dict.fromkeys(paths))


def snapshot(sandbox: Sandbox, paths: list[str]) -> dict[str, str | None]:
    """Current contents of *paths*; None marks a file that does not exist yet."""
    saved: dict[str, str | None] = {}
    for path in paths:
        try:
            target = sandbox.resolve(path)
        except SandboxError:
            continue
        saved[path] = target.read_text(errors="replace") if target.is_file() else None
    return saved


def restore(sandbox: Sandbox, saved: dict[str, str | None]) -> None:
    """Put the workspace back exactly as :func:`snapshot` found it."""
    for path, content in saved.items():
        try:
            target = sandbox.resolve(path)
        except SandboxError:
            continue
        if content is None:
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)


def relevant_files(sandbox: Sandbox, ticket, plan: dict, hints: list[str] | None = None,
                   limit: int = 6, budget: int = 24000) -> str:
    """Existing sources the implementer will likely need, under a char budget.

    *hints* are paths the work journal says prior tickets actually wrote. They
    beat the ticket's own artifact list, which is a guess made before the code
    existed and often names files nobody ever created.

    *budget* is what the caller has left after everything that outranks this
    block; :func:`generate` takes it from the model's context rather than
    guessing, because the files that do not fit here are read as a guess about
    what the ticket needs anyway.
    """
    wanted: list[str] = list(ticket.artifacts) + list(hints or [])
    for component in plan.get("components") or []:
        if isinstance(component, dict) and component.get("path"):
            wanted.append(component["path"])
    chunks, used = [], 0
    seen: set[str] = set()
    for path in wanted:
        path = str(path).lstrip("/")
        if path in seen or len(chunks) >= limit:
            continue
        seen.add(path)
        try:
            text = sandbox.read(path)
        except SandboxError:
            continue
        if not text:
            continue
        slice_ = text[: max(0, budget - used)]
        if not slice_:
            break
        used += len(slice_)
        chunks.append(f"--- {path} ---\n{slice_}")
    return "\n\n".join(chunks) or "(no existing files for this ticket yet)"


def generate(agents, sandbox: Sandbox, state, ticket, tree: str,
             instructions: str = "", failure: str = "", prior: str = "",
             hints: list[str] | None = None, *, allow_edits: bool = True,
             temperature: float | None = None, toolbox: Toolbox | None = None,
             role: str = "", files: int = 6, log: Logger = NULL) -> dict:
    """Ask the right agent for the code that satisfies *ticket*.

    *role* overrides which agent is asked, which is how a repeated failure
    escalates onto a reasoning role; see :func:`~hermes.agents.escalate`.
    *files* is how many existing sources the prompt may show it.

    *toolbox* lets the agent read the rest of the workspace before it answers.
    The sources below are a guess at what the ticket needs, made from a list
    the planner wrote before the code existed; the lookups are how the agent
    checks a name instead of inventing one.

    Blocks are added in reading order, stable first, so a retry can reuse the
    inference server's cached prefix: the project, the ticket and the work
    journal are identical across all attempts at this ticket, and only the
    tail -- the file listing, the current contents, and the failure being
    repaired -- differs between them. Their priorities are a different
    ordering: what survives when the model's context cannot hold all of it.
    The repair instructions and the failure outrank the sources, which outrank
    the journal, which outranks the file listing.
    """
    agent = agents["tester"] if ticket.kind == "test" else agents["implementer"]
    if role:
        agent = escalate(agents, agent, role, log)
    prompt = Prompt(agent, log=log)
    prompt.add("context", context_block(state), required=True)
    prompt.add("ticket", ticket_block(ticket), required=True)
    prompt.add("journal", prior, priority=40, floor=1200)
    prompt.add("tree", f"FILES:\n{tree}", priority=50, floor=400)
    sources = prompt.add("sources", priority=30, floor=2000)
    if role:
        # Swapping the role without saying why buys little: the failure text is
        # the same one the last attempt read, and it argues for the same small
        # patch. Asking a reasoning role to reconsider the approach is the
        # strategy change; the role alone is only who is asked.
        prompt.add("escalation", prompts.ESCALATION, priority=15)
    if instructions:
        prompt.add("repair", f"REPAIR INSTRUCTIONS:\n{instructions}", priority=10)
    if failure:
        prompt.add("failure", f"PREVIOUS ATTEMPT FAILED:\n{failure}",
                   priority=20, floor=800, tail=True)
    prompt.add("closing", prompts.IMPLEMENT, required=True)
    prompt.add("protocol", file_protocol(edits=allow_edits), required=True)
    lookup = Lookup(agent, prompt, toolbox, log=log)
    # Filled last so the room it is given accounts for the repair instructions,
    # the failure and the lookup protocol, all of which outrank it: knowing
    # what broke is worth more than a sixth file the ticket may not touch.
    header = "EXISTING FILES RELEVANT TO THIS TICKET:\n"
    sources.fill(header + relevant_files(sandbox, ticket, state.plan, hints,
                                         limit=files,
                                         budget=max(0, sources.room() - len(header))))
    extra = {} if temperature is None else {"temperature": temperature}
    return lookup.ask(**extra)


def signature_of(verification: Verification | None = None,
                 change: "Change | None" = None, *, prefix: str = "gates") -> str:
    """Name the failure an attempt is handing to the next one.

    A signature is a class of failure, not an instance of one: the ledger
    learns which retry rescues "the build broke", and no two build failures
    share their text. So it is the failing gates, in a fixed order, and never
    their output.

    Gate names are cut at their first colon for the same reason. A ticket's
    own validation gate is named for the ticket, and a signature carrying
    ``T-12`` in it is a signature that matches exactly one ticket in one
    project -- the ledger would fill with rows and never see the same failure
    twice.
    """
    if change is not None and change.empty:
        return "edits:unmatched" if change.unmatched else "output:empty"
    gates = sorted({gate.name.split(":", 1)[0]
                    for gate in verification.failures}) if verification else []
    return f"{prefix}:{'+'.join(gates[:3])}" if gates else f"{prefix}:unknown"


def _stack(state) -> str:
    """The project's primary toolchain, as the ledger buckets it.

    A repair is shaped by its toolchain -- a failing C++ link and a failing
    pytest import are not the same problem in different words -- so what the
    planner named first is kept alongside each attempt.
    """
    technologies = (state.spec or {}).get("technologies") or []
    if isinstance(technologies, str):
        technologies = [technologies]
    first = str(technologies[0]).lower() if technologies else ""
    return (match.group(0) if (match := re.match(r"[a-z+#.]+", first)) else "")[:40]


def execute_ticket(agents, sandbox: Sandbox, state, board, ticket, tree_fn,
                   *, journal=None, max_attempts: int = 3,
                   ladder: Ladder | None = None, ledger=None, project: str = "",
                   allow_edits: bool = True, prefetched: dict | None = None,
                   tool_calls: int = 6, tool_rounds: int = 3,
                   classifier=None, on_apply=None, log: Logger = NULL) -> dict:
    """Run one ticket to a verdict.

    Returns ``{"status": completed|failed, "action": ..., "verification": ...}``.
    The caller owns board transitions beyond the active/review moves here.

    *ladder* says how each attempt is asked; without one every attempt is the
    cheapest ask, which is what the loop did before escalation existed.

    *ledger* is where those asks are remembered and, once the same failure has
    been watched often enough, where the next one is chosen from instead. It
    is advisory in both directions: without one the ladder decides alone, and
    with one it still decides everything the ledger has no evidence about.
    See :mod:`hermes.ledger`.

    *tool_calls* and *tool_rounds* bound the workspace lookups each generation
    and each review may make. Every one of them is a fresh toolbox: the budget
    is what one answer may spend, not what the ticket may.
    """
    ladder = ladder or Ladder()
    ledger = ledger or NullLedger()
    stack = _stack(state)
    def toolbox() -> Toolbox:
        return Toolbox(sandbox, budget=tool_calls, rounds=tool_rounds, log=log)

    board.move(ticket.id, "active", "picked up")
    board.save()
    # Read once: the journal cannot change while this ticket is the one running.
    # An eighth of the implementer's prompt is what history is worth against
    # the code the ticket is about; taking it from the model's context rather
    # than a constant means a bigger model remembers proportionally more.
    prior = journal.block(
        ticket, board,
        budget=max(1000, agents["implementer"].allowance().prompt_chars // 8),
    ) if journal else ""
    hints = journal.suggested_files(ticket, board) if journal else []
    instructions, failure = "", ""
    #: The class of failure the next attempt is being handed, which is what
    #: the ledger keys its record on. Empty on the first attempt: nothing has
    #: gone wrong yet, and a first try is the base rate the rest is judged
    #: against rather than a repair of anything.
    signature = ""
    last_verification: Verification | None = None
    last_change: Change | None = None
    notified = False
    #: True once the workspace holds a tree that cleared the gates and is only
    #: waiting on a review finding. From then on a failing attempt is a
    #: regression against work that was already good, not the ticket's state.
    gates_held = False

    def gate(change: Change) -> Verification:
        nonlocal notified
        if change.dependencies and (state.plan.get("commands") or {}).get("setup"):
            sandbox.install(state.plan["commands"]["setup"], timeout=600)
        for command in change.commands[:3]:
            sandbox.run(command, timeout=300)
        if on_apply is not None and not notified:
            # The gates below are minutes of CPU with the model sitting idle,
            # which is exactly when the factory should start the next ticket.
            notified = True
            on_apply()
        return verify_ticket(
            sandbox, ticket, state.plan, change.written, log=log,
            tests_required=bool(state.metrics.get("tests_passed")),
            classifier=classifier)

    for attempt in range(1, max_attempts + 1):
        # Every record from here down -- model call, sandbox command, gate --
        # belongs to this attempt at this ticket, and says so.
        with scope(ticket=ticket.id, attempt=attempt):
            ticket.attempts += 1
            rung = ladder.rung(attempt)
            # What the ladder would do, unless this exact failure has been
            # watched often enough to know something better -- or something
            # equally good for fewer model calls.
            advice = ledger.advice(signature, rung.spec,
                                   [r.spec for r in ladder.rungs], stack=stack)
            if advice.learned:
                rung = Rung(*advice.rung)
                log.info("ticket.rung_learned", signature=signature,
                         reason=advice.reason, confidence=round(advice.confidence, 2),
                         observations=advice.tried, candidates=rung.candidates,
                         files=rung.files, role=rung.role or "default")
            started = time.time()

            def learned(outcome: str, following: str = "", _rung=rung,
                        _attempt=attempt, _signature=signature) -> None:
                """Tell the ledger how this rung fared against this failure."""
                ledger.record(_signature, _rung.spec, outcome, project=project,
                              stack=stack, kind=ticket.kind, attempt=_attempt,
                              following=following, seconds=time.time() - started)

            log.info("ticket.attempt", title=ticket.title[:60],
                     candidates=rung.candidates, files=rung.files,
                     role=rung.role or "default")
            payloads = _propose(agents, sandbox, state, ticket, tree_fn, instructions,
                                failure, prior, hints, allow_edits=allow_edits,
                                rung=rung,
                                prefetched=prefetched if attempt == 1 else None,
                                toolbox=toolbox, log=log)
            if not payloads:
                failure = "the implementer produced no usable output"
                signature = "output:none"
                learned(FAILED, signature)
                continue

            # A review-driven retry is free to touch files the passing attempt never
            # wrote, so the guard covers what *this* attempt declares it will touch
            # rather than what the last one did; every other file is still the tree
            # that passed and needs no copy. It is taken before anything is applied,
            # so what it captures is the passing content.
            guard = snapshot(
                sandbox, [path for payload in payloads
                          for path in targets_of(payload)]) if gates_held else None

            change, verification = _race(sandbox, payloads, gate, ticket, log)
            last_change = change
            if change is None or change.empty:
                failure = ("\n".join(change.unmatched) if change and change.unmatched
                           else "no files were written; you must return complete "
                                "file contents")
                signature = signature_of(change=change) if change else "output:empty"
                learned(FAILED, signature)
                continue
            last_verification = verification
            if any(g.name == "tests" and g.passed and not g.skipped
                   for g in verification.gates):
                state.metrics["tests_passed"] = True

            if verification.passed:
                verdict = review(agents, ticket, sandbox, change, verification,
                                 prior=prior, toolbox=toolbox(), log=log)
                if verdict.get("approved"):
                    board.move(ticket.id, "review", change.describe())
                    for note in verdict.get("advisory", [])[:3]:
                        ticket.notes.append(f"advisory: {note}")
                    board.save()
                    learned(FIXED)
                    return {"status": "completed", "verification": verification,
                            "change": change, "review": verdict}
                instructions = "Address these review findings:\n" + "\n".join(
                    f"- {b}" for b in verdict["blocking"])
                failure = ""
                gates_held = True
                # The gates passed, so what is being repaired from here is a
                # reviewer's objection and not a broken build: a different
                # failure, asked of a different attempt, and kept apart.
                signature = "review:blocking"
                learned(FAILED, signature)
                continue

            if guard is not None:
                # The gates passed before this attempt, so failing them now is this
                # attempt's regression rather than the ticket's state: put the tree
                # that passed back, so the next attempt starts from the good version
                # instead of the worse one. Triage is deliberately skipped -- the
                # diagnosis is already known, and its verdict could be "decompose"
                # or "rollback", which would throw away the work just recovered.
                # Restoring satisfies none of the review findings: they stay in
                # *instructions*, so the ticket still cannot reach COMPLETED until
                # one change passes the gates and the reviewer both.
                restore(sandbox, guard)
                log.warn("ticket.regression_restored",
                         gates=", ".join(g.name for g in verification.failures))
                failure = (
                    "Your last attempt broke gates that had already passed, so the "
                    "files have been restored to the version that passed them; the "
                    "contents shown above are that restored version. Make the "
                    "smallest change that answers the review findings without "
                    "breaking these gates again:\n" + verification.failure_output())
                if change.unmatched:
                    failure += ("\n\nEDITS THAT DID NOT APPLY:\n"
                                + "\n".join(change.unmatched))
                signature = signature_of(verification, prefix="regression")
                learned(FAILED, signature)
                continue

            diagnosis = triage(agents, ticket, verification,
                               context_block(state, tree_fn()), attempt, log=log)
            action = diagnosis.get("action", "repair")
            if action in ("decompose", "reconsider", "research", "rollback"):
                # The run stopped asking, which says nothing about the rung:
                # recorded so the attempt is not lost, counted for nothing.
                learned(ABANDONED, signature_of(verification))
                return {"status": "failed", "action": action, "diagnosis": diagnosis,
                        "verification": verification, "change": change}
            instructions = diagnosis.get("instructions", "")
            failure = verification.failure_output()
            if change.unmatched:
                failure += "\n\nEDITS THAT DID NOT APPLY:\n" + "\n".join(change.unmatched)
            signature = signature_of(verification)
            learned(FAILED, signature)

    return {"status": "failed", "action": "decompose",
            "diagnosis": {"diagnosis": f"exhausted {max_attempts} attempts",
                          "classification": "implementation", "action": "decompose"},
            "verification": last_verification, "change": last_change}


def _propose(agents, sandbox, state, ticket, tree_fn, instructions, failure, prior,
             hints, *, allow_edits: bool, rung: Rung, prefetched: dict | None,
             toolbox, log: Logger) -> list[dict]:
    """Collect the candidate changes to race through the gates.

    The extra samples are drawn hotter than the first: two candidates from the
    same greedy decoding would be the same candidate twice.
    """
    payloads: list[dict] = []
    if prefetched:
        payloads.append(prefetched)
    while len(payloads) < rung.candidates:
        try:
            payloads.append(generate(
                agents, sandbox, state, ticket, tree_fn(), instructions, failure,
                prior, hints, allow_edits=allow_edits,
                temperature=None if not payloads else 0.6, toolbox=toolbox(),
                role=rung.role, files=rung.files, log=log))
        except Exception as exc:  # model returned nothing usable
            log.error("ticket.generate_failed", ticket=ticket.id, error=str(exc)[:200])
            break
    return payloads


def _race(sandbox: Sandbox, payloads: list[dict], gate, ticket, log: Logger):
    """Apply candidates until one passes the gates; leave the last one in place.

    Verification is the exact, cheap-to-trust judge that a weak local model
    lacks, so the second sample only has to be right often enough to beat the
    cost of running the gates again. A loser's files are rolled back so the
    next candidate starts from the same workspace the first one saw.
    """
    change = verification = None
    for index, payload in enumerate(payloads):
        saved = snapshot(sandbox, targets_of(payload))
        change = apply_change(sandbox, payload, log=log)
        last = index + 1 == len(payloads)
        if change.empty:
            if last:
                break
            continue
        verification = gate(change)
        if verification.passed or last:
            break
        log.info("ticket.candidate_rejected", ticket=ticket.id, candidate=index + 1,
                 gates=verification.report()[:200])
        restore(sandbox, saved)
    return change, verification

