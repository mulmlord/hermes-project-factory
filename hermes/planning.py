"""Intake, specification, architecture and ticket generation."""
from __future__ import annotations

import json
from pathlib import Path

from .log import Logger, NULL
from .tickets import Board, KINDS, Ticket, split_validation
from .workspace import slugify

MAX_TICKET_FIELDS = 400


def intake(agents, idea: str, log: Logger = NULL) -> dict:
    """Turn a one-line idea into a formal, self-contained specification."""
    log.info("intake.start", idea=idea[:100])
    spec = agents["architect"].ask_json(
        f"PROJECT IDEA (the only input the user gave):\n{idea}\n\n"
        "Produce a complete specification. Infer sensible defaults for anything "
        "unstated and record each inference under \"assumptions\" -- you cannot "
        "ask the user. Return JSON:\n"
        "{\n"
        '  "name": "short-kebab-case-project-name",\n'
        '  "objective": "one paragraph: what this software does and for whom",\n'
        '  "deliverables": ["concrete artifact the project must produce"],\n'
        '  "platforms": ["target OS/runtime"],\n'
        '  "technologies": ["language and key libraries, with the language first"],\n'
        '  "functional_requirements": ["testable statement of required behaviour"],\n'
        '  "nonfunctional_requirements": ["performance, security, reliability"],\n'
        '  "constraints": ["hard limits the implementation must respect"],\n'
        '  "risks": ["what could make this fail"],\n'
        '  "unknowns": ["what is not yet known and must be researched"],\n'
        '  "success_criteria": ["measurable, checkable completion condition"],\n'
        '  "assumptions": ["defaults you chose because the idea did not say"],\n'
        '  "optimization_objective": {"metric": "name of an automatically '
        'measurable metric, or empty string if none applies", "direction": '
        '"minimize|maximize", "how_measured": "how a script would measure it"}\n'
        "}\n"
        "Scope it so a small team could build it: ambitious but finishable. "
        "Prefer the standard library and few dependencies."
    )
    spec.setdefault("name", slugify(idea))
    spec["name"] = slugify(spec.get("name") or idea)
    log.info("intake.done", name=spec["name"],
             requirements=len(spec.get("functional_requirements", [])))
    return spec


def architect(agents, spec: dict, research: str, log: Logger = NULL) -> dict:
    """Choose the architecture and the concrete build/test/benchmark commands."""
    log.info("architecture.start")
    plan = agents["architect"].ask_json(
        f"SPECIFICATION:\n{json.dumps(spec, indent=2)[:10000]}\n\n"
        f"RESEARCH CONCLUSIONS:\n{research or '(none)'}\n\n"
        "Design the system and define how it is built and verified. Return JSON:\n"
        "{\n"
        '  "approach": "how you will build this, in a short paragraph",\n'
        '  "architecture": "components and how they fit together",\n'
        '  "components": [{"name": "", "responsibility": "", "path": "src/..."}],\n'
        '  "milestones": [{"name": "", "description": "", "exit_criteria": ""}],\n'
        '  "commands": {"setup": "shell command installing dependencies", '
        '"build": "shell command that compiles/checks, or \'true\' for '
        'interpreted projects", "test": "shell command running the test suite", '
        '"lint": "shell command or empty", "benchmark": "shell command printing '
        'a score, or empty"},\n'
        '  "layout": ["files the project will contain, with one-line purpose"],\n'
        '  "decisions": [{"title": "", "decision": "", "rationale": "", '
        '"alternatives": "", "consequences": ""}]\n'
        "}\n"
        "Commands must be real, runnable, non-interactive shell commands that "
        "work from the project root inside a Linux container. Never use sudo. "
        "The container is a bare base image with nothing but the language "
        "runtime installed. Only the setup command has network access, so it "
        "must install every tool the build, test and lint commands invoke "
        "(for example 'pip install pytest flake8'). Every other command runs "
        "offline."
    )
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
    result = agents["planner"].ask_json(
        f"SPECIFICATION:\n{json.dumps(spec, indent=2)[:8000]}\n\n"
        f"PLAN:\n{json.dumps(plan, indent=2)[:8000]}\n\n"
        f"Break this into at most {max_tickets} tickets. Each ticket must be "
        "small enough for one agent to implement, test and verify in a single "
        "pass -- roughly one component or one coherent feature. Order them so "
        "foundations come first. Return JSON:\n"
        "{\n"
        '  "tickets": [{\n'
        '    "key": "short-unique-key",\n'
        '    "title": "imperative summary",\n'
        '    "objective": "what done looks like, concretely",\n'
        '    "context": "what the implementer needs to know",\n'
        f'    "kind": "one of {"|".join(KINDS)}",\n'
        '    "dependencies": ["key of a ticket that must finish first"],\n'
        '    "acceptance": ["checkable criterion"],\n'
        '    "validation": ["shell command proving it works"],\n'
        '    "artifacts": ["file this ticket creates or changes"],\n'
        '    "priority": 1, "risk": 3, "cost": 2, "info_gain": 2\n'
        "  }]\n"
        "}\n"
        "priority/risk/cost/info_gain are 1-5. Include tickets for tests and "
        "benchmarks where they apply. Do NOT include a README ticket; "
        "documentation is generated at the end. Dependencies must reference "
        "keys in this same list and must not form a cycle."
    )
    return _ingest(board, result.get("tickets") or [], log=log)


def decompose(agents, board: Board, ticket: Ticket, reason: str,
              log: Logger = NULL) -> list[Ticket]:
    """Split a ticket that proved too large into executable children."""
    log.info("planning.decompose", ticket=ticket.id, reason=reason[:120])
    result = agents["planner"].ask_json(
        f"This ticket is too large or keeps failing and must be split.\n\n"
        f"TICKET: {json.dumps({k: getattr(ticket, k) for k in ('id','title','objective','context','acceptance','validation','artifacts')}, indent=2)}\n\n"
        f"WHY IT IS BEING SPLIT:\n{reason[:4000]}\n\n"
        "Return JSON with 2-5 smaller tickets that together fully replace it:\n"
        '{"tickets": [{"key": "", "title": "", "objective": "", "context": "", '
        '"kind": "", "dependencies": ["key of an earlier ticket in this list"], '
        '"acceptance": [""], "validation": [""], "artifacts": [""], '
        '"priority": 1, "risk": 2, "cost": 2, "info_gain": 2}]}\n'
        "Each child must be strictly smaller and independently verifiable."
    )
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
