"""Ticket execution: generate code, apply it, verify, repair, review."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .agents import Prompt, context_block, file_protocol, ticket_block
from .log import Logger, NULL
from .sandbox import Sandbox, SandboxError
from .verify import Verification, review, triage, verify_ticket

#: Files an agent may never write, regardless of what it asks for.
PROTECTED = (".hermes/", ".git/", "hermes.yaml")


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
             rejected=len(change.rejected), unmatched=len(change.unmatched))
    return change


def _apply_edit(sandbox: Sandbox, item: dict, change: Change, log: Logger) -> None:
    """Apply one search/replace edit, or record precisely why it did not apply.

    An edit is only safe when its anchor is unambiguous, so a "find" that is
    absent or appears more than once is refused rather than guessed at. Saying
    which of those happened is what lets the next attempt fix it instead of
    resending the same edit.
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
    occurrences = text.count(find)
    if occurrences == 0:
        change.unmatched.append(
            f"{path}: the text in \"find\" does not appear in the file. Copy it "
            "exactly from the current contents shown to you, or return the whole "
            "file under \"files\".")
        return
    if occurrences > 1:
        change.unmatched.append(
            f"{path}: the text in \"find\" appears {occurrences} times, so the edit "
            "is ambiguous. Include more surrounding lines so it occurs once.")
        return
    sandbox.write(path, text.replace(find, replace, 1))
    if path not in change.written:
        change.written.append(path)


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
             temperature: float | None = None, log: Logger = NULL) -> dict:
    """Ask the right agent for the code that satisfies *ticket*.

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
    prompt = Prompt(agent, log=log)
    prompt.add("context", context_block(state), required=True)
    prompt.add("ticket", ticket_block(ticket), required=True)
    prompt.add("journal", prior, priority=40, floor=1200)
    prompt.add("tree", f"FILES:\n{tree}", priority=50, floor=400)
    sources = prompt.add("sources", priority=30, floor=2000)
    if instructions:
        prompt.add("repair", f"REPAIR INSTRUCTIONS:\n{instructions}", priority=10)
    if failure:
        prompt.add("failure", f"PREVIOUS ATTEMPT FAILED:\n{failure}",
                   priority=20, floor=800, tail=True)
    prompt.add("closing", "Implement this ticket completely. Every file you return "
               "must be complete and runnable. If the ticket needs tests, write "
               "real ones.", required=True)
    prompt.add("protocol", file_protocol(edits=allow_edits), required=True)
    # Filled last so the room it is given accounts for the repair instructions
    # and the failure, which outrank it: knowing what broke is worth more than
    # a sixth file the ticket may not touch.
    header = "EXISTING FILES RELEVANT TO THIS TICKET:\n"
    sources.fill(header + relevant_files(sandbox, ticket, state.plan, hints,
                                         budget=max(0, sources.room() - len(header))))
    extra = {} if temperature is None else {"temperature": temperature}
    return agent.ask_json(prompt.render(), **extra)


def execute_ticket(agents, sandbox: Sandbox, state, board, ticket, tree_fn,
                   *, journal=None, max_attempts: int = 3, candidates: int = 1,
                   allow_edits: bool = True, prefetched: dict | None = None,
                   on_apply=None, log: Logger = NULL) -> dict:
    """Run one ticket to a verdict.

    Returns ``{"status": completed|failed, "action": ..., "verification": ...}``.
    The caller owns board transitions beyond the active/review moves here.
    """
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
            tests_required=bool(state.metrics.get("tests_passed")))

    for attempt in range(1, max_attempts + 1):
        ticket.attempts += 1
        log.info("ticket.attempt", ticket=ticket.id, attempt=attempt,
                 title=ticket.title[:60])
        payloads = _propose(agents, sandbox, state, ticket, tree_fn, instructions,
                            failure, prior, hints, allow_edits=allow_edits,
                            wanted=max(1, candidates) if attempt == 1 else 1,
                            prefetched=prefetched if attempt == 1 else None, log=log)
        if not payloads:
            failure = "the implementer produced no usable output"
            continue

        # A review-driven retry is free to touch files the passing attempt never
        # wrote, so the guard covers what *this* attempt declares it will touch
        # rather than what the last one did; every other file is still the tree
        # that passed and needs no copy. It is taken before anything is applied,
        # so what it captures is the passing content.
        guard = snapshot(sandbox, [path for payload in payloads
                                   for path in targets_of(payload)]) if gates_held else None

        change, verification = _race(sandbox, payloads, gate, ticket, log)
        last_change = change
        if change is None or change.empty:
            failure = ("\n".join(change.unmatched) if change and change.unmatched
                       else "no files were written; you must return complete file contents")
            continue
        last_verification = verification
        if any(g.name == "tests" and g.passed and not g.skipped for g in verification.gates):
            state.metrics["tests_passed"] = True

        if verification.passed:
            verdict = review(agents, ticket, sandbox, change, verification,
                             prior=prior, log=log)
            if verdict.get("approved"):
                board.move(ticket.id, "review", change.describe())
                for note in verdict.get("advisory", [])[:3]:
                    ticket.notes.append(f"advisory: {note}")
                board.save()
                return {"status": "completed", "verification": verification,
                        "change": change, "review": verdict}
            instructions = "Address these review findings:\n" + "\n".join(
                f"- {b}" for b in verdict["blocking"])
            failure = ""
            gates_held = True
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
            log.warn("ticket.regression_restored", ticket=ticket.id,
                     gates=", ".join(g.name for g in verification.failures))
            failure = (
                "Your last attempt broke gates that had already passed, so the "
                "files have been restored to the version that passed them; the "
                "contents shown above are that restored version. Make the "
                "smallest change that answers the review findings without "
                "breaking these gates again:\n" + verification.failure_output())
            if change.unmatched:
                failure += "\n\nEDITS THAT DID NOT APPLY:\n" + "\n".join(change.unmatched)
            continue

        diagnosis = triage(agents, ticket, verification,
                           context_block(state, tree_fn()), attempt, log=log)
        action = diagnosis.get("action", "repair")
        if action in ("decompose", "reconsider", "research", "rollback"):
            return {"status": "failed", "action": action, "diagnosis": diagnosis,
                    "verification": verification, "change": change}
        instructions = diagnosis.get("instructions", "")
        failure = verification.failure_output()
        if change.unmatched:
            failure += "\n\nEDITS THAT DID NOT APPLY:\n" + "\n".join(change.unmatched)

    return {"status": "failed", "action": "decompose",
            "diagnosis": {"diagnosis": f"exhausted {max_attempts} attempts",
                          "classification": "implementation", "action": "decompose"},
            "verification": last_verification, "change": last_change}


def _propose(agents, sandbox, state, ticket, tree_fn, instructions, failure, prior,
             hints, *, allow_edits: bool, wanted: int, prefetched: dict | None,
             log: Logger) -> list[dict]:
    """Collect the candidate changes to race through the gates.

    The extra samples are drawn hotter than the first: two candidates from the
    same greedy decoding would be the same candidate twice.
    """
    payloads: list[dict] = []
    if prefetched:
        payloads.append(prefetched)
    while len(payloads) < wanted:
        try:
            payloads.append(generate(
                agents, sandbox, state, ticket, tree_fn(), instructions, failure,
                prior, hints, allow_edits=allow_edits,
                temperature=None if not payloads else 0.6, log=log))
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

