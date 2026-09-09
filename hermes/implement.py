"""Ticket execution: generate code, apply it, verify, repair, review."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .agents import FILE_PROTOCOL, context_block, ticket_block
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

    @property
    def touched(self) -> list[str]:
        return self.written + self.deleted

    def describe(self) -> str:
        parts = []
        if self.written:
            parts.append(f"wrote {', '.join(self.written[:8])}")
        if self.deleted:
            parts.append(f"deleted {', '.join(self.deleted[:5])}")
        if self.rejected:
            parts.append(f"rejected {', '.join(self.rejected[:5])}")
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
             rejected=len(change.rejected))
    return change


def _protected(path: str) -> bool:
    normalised = path.replace("\\", "/")
    return (normalised.startswith(PROTECTED) or normalised.startswith("../")
            or "/../" in normalised or Path(normalised).is_absolute())


def relevant_files(sandbox: Sandbox, ticket, plan: dict, limit: int = 6,
                   budget: int = 24000) -> str:
    """Existing sources the implementer will likely need, under a char budget."""
    wanted: list[str] = list(ticket.artifacts)
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
             instructions: str = "", failure: str = "", log: Logger = NULL) -> dict:
    """Ask the right agent for the code that satisfies *ticket*."""
    agent = agents["tester"] if ticket.kind == "test" else agents["implementer"]
    existing = relevant_files(sandbox, ticket, state.plan)
    prompt = (
        f"{context_block(state, tree)}\n\n{ticket_block(ticket)}\n\n"
        f"EXISTING FILES RELEVANT TO THIS TICKET:\n{existing}\n\n"
        + (f"REPAIR INSTRUCTIONS:\n{instructions}\n\n" if instructions else "")
        + (f"PREVIOUS ATTEMPT FAILED:\n{failure[:6000]}\n\n" if failure else "")
        + "Implement this ticket completely. Every file you return must be "
        "complete and runnable. If the ticket needs tests, write real ones.\n\n"
        + FILE_PROTOCOL
    )
    return agent.ask_json(prompt)


def execute_ticket(agents, sandbox: Sandbox, state, board, ticket, tree_fn,
                   *, max_attempts: int = 3, log: Logger = NULL) -> dict:
    """Run one ticket to a verdict.

    Returns ``{"status": completed|failed, "action": ..., "verification": ...}``.
    The caller owns board transitions beyond the active/review moves here.
    """
    board.move(ticket.id, "active", "picked up")
    board.save()
    instructions, failure = "", ""
    last_verification: Verification | None = None
    last_change: Change | None = None

    for attempt in range(1, max_attempts + 1):
        ticket.attempts += 1
        log.info("ticket.attempt", ticket=ticket.id, attempt=attempt,
                 title=ticket.title[:60])
        try:
            payload = generate(agents, sandbox, state, ticket, tree_fn(),
                               instructions, failure, log=log)
        except Exception as exc:  # model returned nothing usable
            log.error("ticket.generate_failed", ticket=ticket.id, error=str(exc)[:200])
            failure = f"the implementer produced no usable output: {exc}"
            continue

        change = apply_change(sandbox, payload, log=log)
        last_change = change
        if not change.written and not change.deleted:
            failure = "no files were written; you must return complete file contents"
            continue

        if change.dependencies and (state.plan.get("commands") or {}).get("setup"):
            sandbox.run(state.plan["commands"]["setup"], timeout=600)

        for command in change.commands[:3]:
            sandbox.run(command, timeout=300)

        verification = verify_ticket(
            sandbox, ticket, state.plan, change.written, log=log,
            tests_required=bool(state.metrics.get("tests_passed")))
        last_verification = verification
        if any(g.name == "tests" and g.passed and not g.skipped for g in verification.gates):
            state.metrics["tests_passed"] = True

        if verification.passed:
            verdict = review(agents, ticket, _change_view(sandbox, change),
                             verification, log=log)
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

    return {"status": "failed", "action": "decompose",
            "diagnosis": {"diagnosis": f"exhausted {max_attempts} attempts",
                          "classification": "implementation", "action": "decompose"},
            "verification": last_verification, "change": last_change}


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
