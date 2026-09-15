"""Ticket execution: generate code, apply it, verify, repair, review."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .agents import context_block, file_protocol, ticket_block
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

    Ordered stable-first so a retry can reuse the inference server's cached
    prefix: the project, the ticket and the work journal are identical across
    all attempts at this ticket, and only the tail -- the file listing, the
    current contents, and the failure being repaired -- differs between them.
    """
    agent = agents["tester"] if ticket.kind == "test" else agents["implementer"]
    existing = relevant_files(sandbox, ticket, state.plan, hints)
    prompt = (
        f"{context_block(state)}\n\n{ticket_block(ticket)}\n\n"
        + (f"{prior}\n\n" if prior else "")
        + f"FILES:\n{tree}\n\n"
        + f"EXISTING FILES RELEVANT TO THIS TICKET:\n{existing}\n\n"
        + (f"REPAIR INSTRUCTIONS:\n{instructions}\n\n" if instructions else "")
        + (f"PREVIOUS ATTEMPT FAILED:\n{failure[:6000]}\n\n" if failure else "")
        + "Implement this ticket completely. Every file you return must be "
        "complete and runnable. If the ticket needs tests, write real ones.\n\n"
        + file_protocol(edits=allow_edits)
    )
    extra = {} if temperature is None else {"temperature": temperature}
    return agent.ask_json(prompt, **extra)


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
    prior = journal.block(ticket, board) if journal else ""
    hints = journal.suggested_files(ticket, board) if journal else []
    instructions, failure = "", ""
    last_verification: Verification | None = None
    last_change: Change | None = None
    notified = False

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
            verdict = review(agents, ticket, _change_view(sandbox, change),
                             verification, prior=prior, log=log)
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


def _change_view(sandbox: Sandbox, change: Change, budget: int = 20000) -> str:
    """The written files, truncated, as the reviewer's view of the change."""
    parts, used = [], 0
    for path in change.written:
        text = sandbox.read(path)
        slice_ = text[: max(0, budget - used)]
        if not slice_:
            break
        used += len(slice_)
        parts.append(f"--- {path} ---\n{slice_}")
    if change.deleted:
        parts.append("deleted: " + ", ".join(change.deleted))
    return "\n\n".join(parts) or change.describe()
