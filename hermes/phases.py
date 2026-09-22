"""The nine phases of a project, each a unit the loop can drive on its own.

A phase is a function of the factory: it reads project state, does one
stage's work, and transitions the state machine. It owns no state and
chooses nothing about what runs next -- :class:`hermes.factory.Factory`
keeps the state, the scheduling and the failure recovery, and :data:`PHASES`
is all it needs from here.

Keeping them out of the class is what makes a phase addressable. Bound to it,
no phase could be run, replaced or measured without first standing up a whole
Factory -- router, sandbox, board, repository and all -- while a function
taking the factory can be handed as little as the phase actually touches.
"""
from __future__ import annotations

from . import prompts, workspace
from .evolve import Evolution, objective_from_spec
from .experiment import run_ticket as run_experiment_ticket
from .implement import Change, execute_ticket
from .log import scope
from .planning import _ingest, architect, generate_tickets, intake as specify
from .research import investigate, topics_for
from .tickets import KINDS
from .verify import record_evaluation, scan_unsafe, verify_project


def intake(factory) -> None:
    spec = specify(factory.agents, factory.state.idea, factory.log)
    factory.state.spec = spec
    factory.state.name = spec.get("name", factory.state.name)
    workspace.write_spec(factory.root, spec)
    factory.save("intake")
    factory.commit(f"chore: specify {factory.state.name}\n\n{spec.get('objective','')[:400]}")
    nxt = "RESEARCH" if factory.config.get("research.enabled", True) else "SPECIFICATION"
    factory.state.transition(nxt, "intake complete")


def research(factory) -> None:
    topics = topics_for(factory.agents, factory.context(), factory.state.idea,
                        factory.state.spec,
                        limit=int(factory.config.get("research.max_rounds", 3)),
                        log=factory.log)
    for topic in topics:
        payload = investigate(
            factory.agents, factory.root, topic, factory.context(),
            max_unknowns=int(factory.config.get("research.max_unknowns", 6)),
            log=factory.log, librarian=factory._librarian(),
            db=factory.context_db, classifier=factory.classifier)
        factory._record_decisions(payload.get("decisions", []))
        _queue_research_tickets(factory, payload.get("tickets", []))
        _queue_hypotheses(factory, payload.get("hypotheses", []), topic)
    factory.save("research")
    factory.commit("docs: record research findings and decisions", kind="docs")
    factory.state.transition("SPECIFICATION", f"{len(topics)} research topic(s)")


def _queue_research_tickets(factory, raw: list) -> None:
    _ingest(factory.board, [t for t in raw if isinstance(t, dict)], log=factory.log)


def _queue_hypotheses(factory, raw: list, topic: str) -> None:
    """Turn the round's hypotheses into experiments that will settle them.

    A hypothesis the factory only writes down is a note: it is rendered
    into `docs/research/`, read by nobody, and the unknown it was meant to
    close stays open. Queued as tickets they go through the experiment
    path, and what they measure comes back to the next research round.

    Capped per round, because a round that proposes six claims should not
    push six measurements ahead of the work they exist to inform. The
    cheapest and most informative of them is what the board's own scoring
    picks up first.
    """
    if not factory.config.get("research.experiments", True):
        return
    limit = int(factory.config.get("research.experiments_per_round", 2))
    known = {t.objective for t in factory.board.tickets.values()
             if t.kind == "experiment"}
    fresh = []
    for item in raw:
        statement = str(item.get("statement", ""))[:300] if isinstance(item, dict) else ""
        if not statement or statement in known:
            continue
        known.add(statement)
        fresh.append({
            "title": f"Measure: {statement[:60]}",
            "objective": statement,
            "kind": "experiment",
            "context": f"Hypothesis from research on {topic}.\n"
                       f"Proposed test: {item.get('test', '(none given)')}",
            "acceptance": ["the hypothesis is settled by a measurement rather "
                           "than by assertion",
                           "the experiment records its result under experiments/"],
            # Cheap and high information: an experiment exists to change a
            # decision, and it is worth least once that decision is made.
            "priority": 2, "cost": 2, "risk": 2, "info_gain": 5,
        })
        if len(fresh) >= limit:
            break
    if fresh:
        _queue_research_tickets(factory, fresh)
        factory.board.save()


def specification(factory) -> None:
    plan = architect(factory.agents, factory.state.spec, factory.research_summary(), factory.log)
    factory.state.plan = plan
    factory._record_decisions(plan.get("decisions", []))
    workspace.write_plan(factory.root, plan, factory.board.summary())
    factory.save("specification")
    factory.commit("docs: architecture and build plan", kind="docs")
    factory.state.transition("PLANNING", "architecture chosen")


def planning(factory) -> None:
    generate_tickets(factory.agents, factory.board, factory.state.spec, factory.state.plan,
                     max_tickets=int(factory.config.get("planning.max_tickets", 40)),
                     log=factory.log)
    factory.board.unblock()
    workspace.write_plan(factory.root, factory.state.plan, factory.board.summary())
    factory.save("planning")
    factory.commit(f"chore: plan {len(factory.board.tickets)} tickets")
    _publish_board(factory)
    factory.state.transition("IMPLEMENTATION", f"{len(factory.board.tickets)} tickets")


def _publish_board(factory) -> None:
    """Create the repository as soon as there is a board worth watching.

    The issue projection can only act once a remote exists, and the remote
    used to arrive in the release phase -- by which point every ticket is
    already closed and there is nothing left to observe. So the repository
    is created here, where the board first exists, and the code is held
    back until a ticket has passed its gates: repositories are private by
    default, and a scaffold nobody has validated is not worth showing.
    """
    if not (factory.config.get("github.enabled", False)
            and factory.config.get("github.early", True)):
        return
    url = factory.github.create(
        factory.root, factory.config.get("github.name") or factory.state.name,
        (factory.state.spec.get("objective") or factory.state.idea)[:200])
    if url:
        factory.state.metrics["github_url"] = url


def _push_first_green(factory) -> None:
    """Push the code once something has actually passed, and only once."""
    if not factory.config.get("github.enabled", False):
        return
    if factory.state.metrics.get("github_pushed"):
        return
    if not factory.state.metrics.get("github_url"):
        return
    if factory.github.push(factory.root):
        factory.state.metrics["github_pushed"] = True


def implementation(factory) -> None:
    """Execute one ticket per call so the outer loop can checkpoint."""
    factory.board.unblock()
    ticket = factory.board.next_ticket()
    if ticket is None:
        if factory.board.by_status("active", "blocked"):
            factory._recover_stalled()
            return
        factory.state.transition("VALIDATION", "no ready tickets")
        return

    # Everything below belongs to this ticket, down to the model call that
    # generated a patch and the command that rejected it.
    with scope(ticket=ticket.id):
        if ticket.kind == "experiment":
            # An experiment is not implemented and reviewed, it is run and
            # measured, so it takes its own path to the same verdict shape.
            outcome = _run_experiment(factory, ticket)
        elif factory.openhands.should_handle(ticket) and _run_openhands(factory, ticket):
            return
        else:
            outcome = execute_ticket(
                factory.agents, factory.sandbox, factory.state, factory.board, ticket, factory.tree,
                journal=factory.journal,
                max_attempts=int(factory.config.get("execution.max_ticket_attempts", 3)),
                ladder=factory.ladder,
                ledger=factory.ledger,
                project=factory.state.name,
                allow_edits=bool(factory.config.get("execution.edit_protocol", True)),
                prefetched=factory._take_prefetch(ticket),
                tool_calls=int(factory.config.get("execution.tool_calls", 6)),
                tool_rounds=int(factory.config.get("execution.tool_rounds", 3)),
                db=factory.context_db,
                classifier=factory.classifier,
                mutation=factory.config.get("verification.mutation") or {},
                on_apply=lambda: factory._prefetch_next(ticket),
                log=factory.log)

        if outcome["status"] == "completed":
            factory.board.move(ticket.id, "completed",
                           outcome.get("note", "all gates passed"))
            factory.state.clear_failure(ticket.id)
            record_evaluation(factory.root, ticket.id, outcome["verification"],
                              {"review": outcome.get("review", {})})
            sha = factory.commit(f"{_commit_kind(ticket.kind)}: {ticket.title}\n\n"
                                 f"Ticket {ticket.id}. {outcome['change'].summary}",
                                 kind=_commit_kind(ticket.kind))
            factory.journal.record(ticket, outcome, commit=sha or "")
            factory.log.info("ticket.completed",
                             progress=f"{factory.board.progress():.0%}")
            _push_first_green(factory)
        else:
            factory._handle_ticket_failure(ticket, outcome)
        factory.save(f"ticket-{ticket.id}")


def _run_experiment(factory, ticket) -> dict:
    """Settle an experiment ticket by measurement instead of implementation.

    Until this existed a ``kind: "experiment"`` ticket was handed to the
    implementer, which answered acceptance criteria that describe a
    measurement by writing code that asserts it -- the one thing an
    experiment must not do.
    """
    factory.board.move(ticket.id, "active", "running experiment")
    factory.board.save()
    outcome = run_experiment_ticket(
        factory._lab(), ticket, tree=factory.tree(), toolbox=factory._toolbox(),
        version=factory.repo.head() if factory.repo.available else "")
    record = outcome["record"]
    outcome["note"] = f"{record.id}: {record.verdict}"
    factory.log.info("ticket.experiment", experiment=record.id,
                     verdict=record.verdict, measured=len(record.measurements))
    return outcome


def _run_openhands(factory, ticket) -> bool:
    factory.board.move(ticket.id, "active", "delegated to OpenHands")
    factory.board.save()
    result = factory.openhands.run_ticket(factory.root, ticket, factory.context())
    if not result["ok"]:
        factory.board.move(ticket.id, "ready", "OpenHands failed; native agent will retry")
        factory.board.save()
        return False
    from .verify import verify_ticket
    verification = verify_ticket(factory.sandbox, ticket, factory.state.plan, [],
                                 factory.log, classifier=factory.classifier)
    if verification.passed:
        factory.board.move(ticket.id, "completed", "OpenHands + gates passed")
        sha = factory.commit(f"{_commit_kind(ticket.kind)}: {ticket.title}\n\n"
                             f"Ticket {ticket.id}, implemented via OpenHands.",
                             kind=_commit_kind(ticket.kind))
        # OpenHands edits the workspace itself, so the commit is the only
        # record of which files it actually touched.
        factory.journal.record(
            ticket,
            {"status": "completed", "verification": verification,
             "change": Change(summary="implemented by the OpenHands agent",
                              written=factory.repo.files_in())},
            commit=sha or "")
        factory.save(f"ticket-{ticket.id}")
        return True
    factory.board.move(ticket.id, "ready", "OpenHands output failed gates")
    factory.board.save()
    return False


def validation(factory) -> None:
    verification = verify_project(factory.sandbox, factory.state.plan,
                                  classifier=factory.classifier, log=factory.log)
    record_evaluation(factory.root, "project-validation", verification)
    factory.log.info("validation", passed=verification.passed,
                     gates=len(verification.gates))
    if verification.passed:
        objective = objective_from_spec(factory.state.spec, factory.state.plan)
        if objective and factory.config.get("evolution.enabled", True):
            factory.state.transition("OPTIMIZATION", "measurable objective present")
        else:
            factory.state.transition("DOCUMENTATION", "validation passed")
        return
    _repair_from_validation(factory, verification)


def _repair_from_validation(factory, verification) -> None:
    count = factory.state.record_failure("validation")
    if count > 4:
        factory.state.transition("DOCUMENTATION",
                                 "validation still failing; documenting known defects")
        factory.state.notes.append(
            "Validation did not converge. Known defects:\n" + verification.report())
        return
    fix = factory.agents["triager"].ask_json(
        prompts.repair_tickets(factory.context(), verification.report(),
                               verification.failure_output()),
        schema=prompts.tickets_schema(KINDS))
    created = _ingest(factory.board, fix.get("tickets") or [], log=factory.log)
    if not created:
        factory.board.add(title="Fix failing project validation", priority=1,
                          objective="Make the build and test suite pass.",
                          context=verification.failure_output(3000),
                          acceptance=["build succeeds", "all tests pass"],
                          validation=[(factory.state.plan.get("commands")
                                       or {}).get("test", "true")],
                          status="ready")
        factory.board.save()
    factory.state.transition("IMPLEMENTATION", f"validation repair round {count}")


def optimization(factory) -> None:
    objective = objective_from_spec(factory.state.spec, factory.state.plan)
    if objective is None:
        factory.state.transition("DOCUMENTATION", "no measurable objective")
        return
    targets = _evolution_targets(factory)
    if not targets:
        factory.state.transition("DOCUMENTATION", "no optimisation targets found")
        return
    engine = Evolution(factory.sandbox, factory.agents, factory.state, objective, targets,
                       factory.config.get("evolution", {}), factory.log,
                       governor=factory.governor)
    best = engine.run(
        max_generations=int(factory.config.get("evolution.max_generations", 20)),
        patience=int(factory.config.get("evolution.patience", 5)))
    if best:
        factory.state.metrics["optimization"] = {
            "metric": objective.metric, "direction": objective.direction,
            "baseline": engine.history[0].score, "best": best.score,
            "generations": len(engine.history),
        }
        (factory.root / "docs" / "generated" / "optimization.md").write_text(
            f"# Optimisation report\n\n```\n{engine.report()}\n```\n")
        factory.commit(f"perf: evolve {objective.metric} to {best.score}", kind="perf")
    factory.save("optimization")
    factory.state.transition(
        "VALIDATION" if best and best.generation else "DOCUMENTATION",
        "evolution finished")


def _evolution_targets(factory) -> list[str]:
    """Source files that actually influence the benchmark."""
    result = factory.agents["architect"].ask_json(
        prompts.evolution_targets(
            factory.context(),
            (factory.state.plan.get("commands") or {}).get("benchmark")),
        schema=prompts.TARGETS)
    return [f for f in (result.get("files") or [])
            if isinstance(f, str) and (factory.root / f).is_file()][:4]


def documentation(factory) -> None:
    from .docs import write_all
    write_all(factory.agents, factory.sandbox, factory.state, factory.board, factory.log)
    factory.save("documentation")
    factory.commit("docs: generate README and project documentation", kind="docs")
    factory.state.transition("RELEASE", "documentation generated")


def release(factory) -> None:
    from .release import validate_clean_room
    verification, report = validate_clean_room(factory.sandbox, factory.state, factory.log)
    record_evaluation(factory.root, "release-validation", verification)
    (factory.root / "docs" / "generated" / "release-validation.md").write_text(report)
    criteria = completion_criteria(factory, verification)
    factory.save("release")
    if all(criteria.values()):
        version = "v0.1.0"
        factory.commit("chore: release validation passed", kind="chore")
        if factory.repo.available:
            factory.repo.tag(version, f"{factory.state.name} {version}")
        url = factory.github.create_and_push(
            factory.root, factory.config.get("github.name") or factory.state.name,
            (factory.state.spec.get("objective") or factory.state.idea)[:200])
        if url:
            factory.state.metrics["github_url"] = url
            factory.github.release(
                factory.root, version, f"{factory.state.name} {version}",
                (factory.root / "README.md").read_text()[:5000])
        factory.state.metrics["completion"] = criteria
        factory.state.transition("COMPLETED", "all completion criteria satisfied")
        return
    failed = [k for k, v in criteria.items() if not v]
    count = factory.state.record_failure("release")
    factory.log.warn("release.blocked", failed=", ".join(failed), attempt=count)
    if count > 2:
        factory.state.metrics["completion"] = criteria
        factory.state.transition("FAILED", f"release criteria unmet: {failed}")
        return
    factory.state.transition("IMPLEMENTATION", f"release blocked on {failed}")


def completion_criteria(factory, verification) -> dict[str, bool]:
    """Section 25: completion is more than an empty board."""
    gates = {g.name: (g.passed or g.skipped) for g in verification.gates}
    readme = (factory.root / "README.md").read_text() if (factory.root / "README.md").is_file() else ""
    outstanding = [t for t in factory.board.by_status("ready", "active", "blocked")]
    critical = [t for t in factory.board.by_status("abandoned") if t.priority <= 2]
    metrics = factory.state.metrics.get("optimization")
    return {
        "requirements_satisfied": not outstanding,
        "build_succeeds": gates.get("build", False),
        "tests_pass": gates.get("tests", False),
        "integration_tests_pass": gates.get("tests", False),
        "no_critical_defects": not critical,
        "performance_satisfied": metrics is None or metrics.get("best") is not None,
        "documentation_exists": len(readme) > 400 and "## " in readme,
        "reproducible": gates.get("clean-room", False),
        "secure": not scan_unsafe(factory.root),
    }


def _commit_kind(ticket_kind: str) -> str:
    return {"test": "test", "documentation": "docs", "optimization": "perf",
            "refactor": "refactor", "research": "docs", "experiment": "chore"}.get(
        ticket_kind, "feat")


#: The phase each project state runs, and the whole of what the loop needs
#: from this module. A state missing from it is one the loop never drives:
#: COMPLETED and FAILED are terminal, and the driver stops before looking.
PHASES = {
    "INTAKE": intake,
    "RESEARCH": research,
    "SPECIFICATION": specification,
    "PLANNING": planning,
    "IMPLEMENTATION": implementation,
    "VALIDATION": validation,
    "OPTIMIZATION": optimization,
    "DOCUMENTATION": documentation,
    "RELEASE": release,
}
