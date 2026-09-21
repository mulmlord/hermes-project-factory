"""Intake, specification, architecture and ticket generation."""
from __future__ import annotations

import json
from pathlib import Path

from . import prompts
from .log import Logger, NULL
from .tickets import Board, KINDS, Ticket, split_validation
from .workspace import slugify

MAX_TICKET_FIELDS = 400


def intake(agents, idea: str, log: Logger = NULL) -> dict:
    """Turn a one-line idea into a formal, self-contained specification."""
    log.info("intake.start", idea=idea[:100])
    spec = agents["architect"].ask_json(prompts.spec_from_idea(idea))
    spec.setdefault("name", slugify(idea))
    spec["name"] = slugify(spec.get("name") or idea)
    log.info("intake.done", name=spec["name"],
             requirements=len(spec.get("functional_requirements", [])))
    return spec


def architect(agents, spec: dict, research: str, log: Logger = NULL) -> dict:
    """Choose the architecture and the concrete build/test/benchmark commands."""
    log.info("architecture.start")
    plan = agents["architect"].ask_json(prompts.system_design(
        json.dumps(spec, indent=2)[:10000], research))
    plan.setdefault("commands", {})
    for key, fallback in (("build", "true"), ("test", "true"), ("setup", "true")):
        if not plan["commands"].get(key):
            plan["commands"][key] = fallback
    log.info("architecture.done",
             components=len(plan.get("components", [])),
             test=plan["commands"].get("test"))
    return plan


def generate_tickets(agents, board: Board, spec: dict, plan: dict, *,
                     max_tickets: int = 40, log: Logger = NULL) -> list[Ticket]:
    """Decompose the plan into an ordered, dependency-aware ticket set."""
    log.info("planning.tickets")
    result = agents["planner"].ask_json(prompts.ticket_breakdown(
        json.dumps(spec, indent=2)[:8000], json.dumps(plan, indent=2)[:8000],
        max_tickets, "|".join(KINDS)))
    return _ingest(board, result.get("tickets") or [], log=log)


def decompose(agents, board: Board, ticket: Ticket, reason: str,
              log: Logger = NULL) -> list[Ticket]:
    """Split a ticket that proved too large into executable children."""
    log.info("planning.decompose", ticket=ticket.id, reason=reason[:120])
    fields = ("id", "title", "objective", "context", "acceptance",
              "validation", "artifacts")
    result = agents["planner"].ask_json(prompts.ticket_split(
        json.dumps({k: getattr(ticket, k) for k in fields}, indent=2),
        reason[:4000]))
    children = _ingest(board, result.get("tickets") or [], parent=ticket,
                       depth=ticket.depth + 1, log=log)
    if children:
        # Anything that depended on the parent now depends on the last child.
        terminal = children[-1].id
        for other in board.tickets.values():
            if other.id not in {c.id for c in children} and ticket.id in other.dependencies:
                other.dependencies = [terminal if d == ticket.id else d
                                      for d in other.dependencies]
        board.move(ticket.id, "abandoned", f"split into {', '.join(c.id for c in children)}")
    return children


def _ingest(board: Board, raw: list, *, parent: Ticket | None = None, depth: int = 0,
            log: Logger = NULL) -> list[Ticket]:
    """Create tickets from model output, mapping author keys to real IDs."""
    key_to_id: dict[str, str] = {}
    created: list[Ticket] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("title"):
            continue
        commands, prose = split_validation(_strings(item.get("validation")))
        if prose:
            log.debug("planning.prose_validation", title=str(item["title"])[:60],
                      dropped=len(prose))
        ticket = board.add(
            title=str(item["title"])[:MAX_TICKET_FIELDS],
            objective=str(item.get("objective", ""))[:2000],
            context=str(item.get("context", ""))[:4000],
            kind=item.get("kind", "implementation"),
            acceptance=_strings(item.get("acceptance")) + prose,
            validation=commands,
            artifacts=_strings(item.get("artifacts")),
            priority=_clamp(item.get("priority"), 3),
            risk=_clamp(item.get("risk"), 3),
            cost=_clamp(item.get("cost"), 3),
            info_gain=_clamp(item.get("info_gain"), 3),
            depth=depth,
            parent=parent.id if parent else None,
        )
        key = str(item.get("key") or item.get("id") or ticket.title)
        key_to_id[key] = ticket.id
        key_to_id[ticket.title] = ticket.id
        created.append(ticket)

    for ticket, item in zip(created, [i for i in raw if isinstance(i, dict) and i.get("title")]):
        deps = []
        for dep in _strings(item.get("dependencies")):
            resolved = key_to_id.get(dep) or (dep if dep in board.tickets else None)
            if resolved and resolved != ticket.id:
                deps.append(resolved)
        ticket.dependencies = deps

    _break_cycles(board, log)
    board.unblock()
    board.save()
    log.info("planning.tickets_created", count=len(created))
    return created


def _break_cycles(board: Board, log: Logger = NULL) -> None:
    """Cut the last edge of any dependency cycle so the scheduler cannot hang."""
    for _ in range(20):
        cycles = board.cycles()
        if not cycles:
            return
        cycle = cycles[0]
        offender, victim = cycle[-2], cycle[-1]
        board.tickets[offender].dependencies = [
            d for d in board.tickets[offender].dependencies if d != victim
        ]
        log.warn("planning.cycle_broken", removed=f"{offender}->{victim}")


def _strings(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v)[:MAX_TICKET_FIELDS] for v in value if str(v).strip()]


def _clamp(value, default: int) -> int:
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return default
