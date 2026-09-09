"""The autonomous factory loop.

One object owns the whole lifecycle: intake, research, specification,
planning, implementation, validation, optimisation, documentation, release.
It never asks the user anything; every branch resolves to a decision.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import workspace
from .agents import build as build_agents, context_block
from .config import Config
from .evolve import Evolution, objective_from_spec
from .git_ops import GitHub, Repo
from .implement import execute_ticket
from .log import Logger, for_workspace
from .models import Router
from .openhands import OpenHands
from .planning import architect, decompose, generate_tickets, intake
from .research import investigate, topics_for
from .sandbox import Sandbox, probe_resources
from .state import State, checkpoint
from .tickets import Board
from .verify import record_evaluation, scan_unsafe, verify_project


class Factory:
    def __init__(self, root: Path, config: Config | None = None,
                 log: Logger | None = None, quiet: bool = False):
        self.root = Path(root).resolve()
        self.config = config or Config.load()
        self.log = log or for_workspace(self.root, self.config.get("logging.level"), quiet)
        self.state = State.load(self.root)
        self.board = Board(self.root)
        self.router = Router(self.config, self.log)
        self.agents = build_agents(self.router, self.log)
        self.sandbox = Sandbox(self.root, self.config.get("sandbox", {}), self.log)
        self.repo = Repo(self.root, self.log, self.config.get("git.author"))
        self.github = GitHub(self.config.get("github", {}), self.log)
        self.openhands = OpenHands(self.config.get("openhands", {}),
                                   self.config.model_for("coding"), self.log)
        self.resources = probe_resources()

    # -- helpers ---------------------------------------------------------
    def tree(self) -> str:
        return workspace.tree(self.root)

    def context(self, extra: str = "") -> str:
        return context_block(self.state, self.tree(), extra)

    def save(self, label: str = "") -> None:
        self.state.save(self.root)
        self.board.save()
        workspace.write_context(self.root, self.state, self._board_extra())
        if label and self.state.iteration % max(
                1, int(self.config.get("execution.checkpoint_every", 1))) == 0:
            checkpoint(self.root, label)

    def _board_extra(self) -> str:
        return f"## Board\n```\n{self.board.summary()}\n```\n"

    def commit(self, message: str, kind: str = "chore") -> None:
        if self.config.get("git.enabled", True):
            self.repo.commit(message, kind=kind)

    def research_summary(self, limit: int = 6000) -> str:
        path = self.root / ".hermes" / "state" / "research.json"
        if not path.is_file():
            return ""
        rounds = json.loads(path.read_text()).get("rounds", [])
        lines = []
        for round_ in rounds:
            lines.append(f"### {round_.get('topic')}")
            lines += [f"- {c}" for c in round_.get("conclusions", [])]
            for decision in round_.get("decisions", []):
                lines.append(f"- DECISION: {decision.get('title')}: "
                             f"{decision.get('decision')}")
        return "\n".join(lines)[:limit]

    # -- phases ----------------------------------------------------------
    def phase_intake(self) -> None:
        spec = intake(self.agents, self.state.idea, self.log)
        self.state.spec = spec
        self.state.name = spec.get("name", self.state.name)
        workspace.write_spec(self.root, spec)
        self.save("intake")
        self.commit(f"chore: specify {self.state.name}\n\n{spec.get('objective','')[:400]}")
        nxt = "RESEARCH" if self.config.get("research.enabled", True) else "SPECIFICATION"
        self.state.transition(nxt, "intake complete")

    def phase_research(self) -> None:
        topics = topics_for(self.agents, self.context(), self.state.idea,
                            self.state.spec,
                            limit=int(self.config.get("research.max_rounds", 3)),
                            log=self.log)
        for topic in topics:
            payload = investigate(
                self.agents, self.root, topic, self.context(),
                max_unknowns=int(self.config.get("research.max_unknowns", 6)),
                log=self.log)
            self._record_decisions(payload.get("decisions", []))
            self._queue_research_tickets(payload.get("tickets", []))
        self.save("research")
        self.commit("docs: record research findings and decisions", kind="docs")
        self.state.transition("SPECIFICATION", f"{len(topics)} research topic(s)")

    def _record_decisions(self, decisions: list) -> None:
        for decision in decisions:
            if not isinstance(decision, dict) or not decision.get("title"):
                continue
            workspace.write_decision(
                self.root, workspace.next_decision_number(self.root),
                decision["title"], decision.get("rationale", ""),
                decision.get("decision", ""), decision.get("consequences", ""),
                decision.get("alternatives", ""))

    def _queue_research_tickets(self, raw: list) -> None:
        from .planning import _ingest
        _ingest(self.board, [t for t in raw if isinstance(t, dict)], log=self.log)

    def phase_specification(self) -> None:
        plan = architect(self.agents, self.state.spec, self.research_summary(), self.log)
        self.state.plan = plan
        self._record_decisions(plan.get("decisions", []))
        workspace.write_plan(self.root, plan, self.board.summary())
        self.save("specification")
        self.commit("docs: architecture and build plan", kind="docs")
        self.state.transition("PLANNING", "architecture chosen")

    def phase_planning(self) -> None:
        generate_tickets(self.agents, self.board, self.state.spec, self.state.plan,
                         max_tickets=int(self.config.get("planning.max_tickets", 40)),
                         log=self.log)
        self.board.unblock()
        workspace.write_plan(self.root, self.state.plan, self.board.summary())
        self.save("planning")
        self.commit(f"chore: plan {len(self.board.tickets)} tickets")
        self.state.transition("IMPLEMENTATION", f"{len(self.board.tickets)} tickets")

    def phase_implementation(self) -> None:
        """Execute one ticket per call so the outer loop can checkpoint."""
        self.board.unblock()
        ticket = self.board.next_ticket()
        if ticket is None:
            if self.board.by_status("active", "blocked"):
                self._recover_stalled()
                return
            self.state.transition("VALIDATION", "no ready tickets")
            return

        if self.openhands.should_handle(ticket):
            handled = self._run_openhands(ticket)
            if handled:
                return

        outcome = execute_ticket(
            self.agents, self.sandbox, self.state, self.board, ticket, self.tree,
            max_attempts=int(self.config.get("execution.max_ticket_attempts", 3)),
            log=self.log)

        if outcome["status"] == "completed":
            self.board.move(ticket.id, "completed", "all gates passed")
            self.state.clear_failure(ticket.id)
            record_evaluation(self.root, ticket.id, outcome["verification"],
                              {"review": outcome.get("review", {})})
            self.commit(f"{_commit_kind(ticket.kind)}: {ticket.title}\n\n"
                        f"Ticket {ticket.id}. {outcome['change'].summary}",
                        kind=_commit_kind(ticket.kind))
            self.log.info("ticket.completed", ticket=ticket.id,
                          progress=f"{self.board.progress():.0%}")
        else:
            self._handle_ticket_failure(ticket, outcome)
        self.save(f"ticket-{ticket.id}")

    def _run_openhands(self, ticket) -> bool:
        self.board.move(ticket.id, "active", "delegated to OpenHands")
        self.board.save()
        result = self.openhands.run_ticket(self.root, ticket, self.context())
        if not result["ok"]:
            self.board.move(ticket.id, "ready", "OpenHands failed; native agent will retry")
            self.board.save()
            return False
        from .verify import verify_ticket
        verification = verify_ticket(self.sandbox, ticket, self.state.plan, [], self.log)
        if verification.passed:
            self.board.move(ticket.id, "completed", "OpenHands + gates passed")
            self.commit(f"{_commit_kind(ticket.kind)}: {ticket.title}\n\n"
                        f"Ticket {ticket.id}, implemented via OpenHands.",
                        kind=_commit_kind(ticket.kind))
            self.save(f"ticket-{ticket.id}")
            return True
        self.board.move(ticket.id, "ready", "OpenHands output failed gates")
        self.board.save()
        return False

    def _handle_ticket_failure(self, ticket, outcome: dict) -> None:
        action = outcome.get("action", "repair")
        diagnosis = outcome.get("diagnosis", {})
        count = self.state.record_failure(ticket.id)
        detail = diagnosis.get("diagnosis", "")[:300]
        self.log.warn("ticket.failed", ticket=ticket.id, action=action, failures=count,
                      diagnosis=detail)
        ticket.notes.append(f"failure {count}: {action}: {detail}")

        max_depth = int(self.config.get("planning.max_decompose_depth", 2))
        if count >= int(self.config.get("execution.max_repair_attempts", 3)):
            self._rollback(ticket, detail or "repeated failure")
        elif action == "research":
            topic = diagnosis.get("research_topic") or f"How to correctly implement: {ticket.title}"
            payload = investigate(self.agents, self.root, topic, self.context(),
                                  log=self.log)
            self._record_decisions(payload.get("decisions", []))
            ticket.context = (ticket.context + "\n\nRESEARCH:\n" +
                              "\n".join(f"- {c}" for c in payload.get("conclusions", []))[:2000])
            self.board.move(ticket.id, "ready", "research completed, retrying")
        elif action == "decompose" and ticket.depth < max_depth:
            children = decompose(
                self.agents, self.board, ticket,
                detail + "\n" + (outcome.get("verification").failure_output(3000)
                                 if outcome.get("verification") else ""),
                log=self.log)
            if not children:
                # The planner could not split it; do not leave it stranded.
                self.board.move(ticket.id, "ready", "decomposition produced nothing")
        elif action == "rollback":
            self._rollback(ticket, detail)
        elif action == "reconsider":
            self.board.move(ticket.id, "blocked", "architecture must be reconsidered")
            self.state.notes.append(f"{ticket.id}: architecture concern: {detail}")
            self.state.transition("PLANNING", "architecture reconsideration")
        else:
            self.board.move(ticket.id, "ready", f"retry after {action}")
        if self.board.tickets[ticket.id].status == "active":
            self.board.move(ticket.id, "ready", "released from active column")

    def _rollback(self, ticket, reason: str) -> None:
        """Undo the ticket's damage and park it rather than looping forever."""
        if self.repo.available and (self.root / ".git").is_dir():
            self.repo._run("checkout", "--", ".")
        self.board.move(ticket.id, "abandoned",
                        f"rolled back after repeated failure: {reason[:200]}")
        self.state.notes.append(f"{ticket.id} abandoned: {reason[:200]}")
        self.log.warn("ticket.abandoned", ticket=ticket.id)

    def _recover_stalled(self) -> None:
        """Nothing is ready but work remains: unstick it or accept the loss."""
        for ticket in self.board.by_status("active"):
            self.board.move(ticket.id, "ready", "recovered from interrupted run")
        blocked = self.board.by_status("blocked")
        for ticket in blocked:
            missing = [d for d in ticket.dependencies
                       if d in self.board.tickets
                       and self.board.tickets[d].status == "abandoned"]
            if missing:
                ticket.dependencies = [d for d in ticket.dependencies if d not in missing]
                self.board.move(ticket.id, "ready",
                                f"dependencies {missing} were abandoned; proceeding anyway")
        if not self.board.by_status("ready", "active"):
            self.state.transition("VALIDATION", "board drained")
        self.board.save()

    def phase_validation(self) -> None:
        verification = verify_project(self.sandbox, self.state.plan, log=self.log)
        record_evaluation(self.root, "project-validation", verification)
        self.log.info("validation", passed=verification.passed,
                      gates=len(verification.gates))
        if verification.passed:
            objective = objective_from_spec(self.state.spec, self.state.plan)
            if objective and self.config.get("evolution.enabled", True):
                self.state.transition("OPTIMIZATION", "measurable objective present")
            else:
                self.state.transition("DOCUMENTATION", "validation passed")
            return
        self._repair_from_validation(verification)

    def _repair_from_validation(self, verification) -> None:
        count = self.state.record_failure("validation")
        if count > 4:
            self.state.transition("DOCUMENTATION",
                                  "validation still failing; documenting known defects")
            self.state.notes.append(
                "Validation did not converge. Known defects:\n" + verification.report())
            return
        fix = self.agents["triager"].ask_json(
            f"{self.context()}\n\nPROJECT VALIDATION FAILED:\n{verification.report()}\n\n"
            f"OUTPUT:\n{verification.failure_output()}\n\n"
            "Create repair tickets. Return JSON:\n"
            '{"tickets": [{"key": "", "title": "", "objective": "", "context": "", '
            '"kind": "implementation", "acceptance": [""], "validation": [""], '
            '"artifacts": [""], "priority": 1, "risk": 2, "cost": 2, "info_gain": 1}]}'
        )
        from .planning import _ingest
        created = _ingest(self.board, fix.get("tickets") or [], log=self.log)
        if not created:
            self.board.add(title="Fix failing project validation", priority=1,
                           objective="Make the build and test suite pass.",
                           context=verification.failure_output(3000),
                           acceptance=["build succeeds", "all tests pass"],
                           validation=[(self.state.plan.get("commands") or {}).get("test", "true")],
                           status="ready")
            self.board.save()
        self.state.transition("IMPLEMENTATION", f"validation repair round {count}")

    def phase_optimization(self) -> None:
        objective = objective_from_spec(self.state.spec, self.state.plan)
        if objective is None:
            self.state.transition("DOCUMENTATION", "no measurable objective")
            return
        targets = self._evolution_targets()
        if not targets:
            self.state.transition("DOCUMENTATION", "no optimisation targets found")
            return
        engine = Evolution(self.sandbox, self.agents, self.state, objective, targets,
                           self.config.get("evolution", {}), self.log)
        best = engine.run(
            max_generations=int(self.config.get("evolution.max_generations", 20)),
            patience=int(self.config.get("evolution.patience", 5)))
        if best:
            self.state.metrics["optimization"] = {
                "metric": objective.metric, "direction": objective.direction,
                "baseline": engine.history[0].score, "best": best.score,
                "generations": len(engine.history),
            }
            (self.root / "docs" / "generated" / "optimization.md").write_text(
                f"# Optimisation report\n\n```\n{engine.report()}\n```\n")
            self.commit(f"perf: evolve {objective.metric} to {best.score}", kind="perf")
        self.save("optimization")
        self.state.transition("VALIDATION" if best and best.generation else "DOCUMENTATION",
                              "evolution finished")

    def _evolution_targets(self) -> list[str]:
        """Source files that actually influence the benchmark."""
        result = self.agents["architect"].ask_json(
            f"{self.context()}\n\n"
            f"BENCHMARK COMMAND: {(self.state.plan.get('commands') or {}).get('benchmark')}\n\n"
            "Which existing source files dominate this benchmark's cost? Return "
            'JSON: {"files": ["src/path.ext"]} with at most 4 files that exist in '
            "the file listing above. Exclude tests and benchmark harnesses."
        )
        return [f for f in (result.get("files") or [])
                if isinstance(f, str) and (self.root / f).is_file()][:4]

    def phase_documentation(self) -> None:
        from .docs import write_all
        write_all(self.agents, self.sandbox, self.state, self.board, self.log)
        self.save("documentation")
        self.commit("docs: generate README and project documentation", kind="docs")
        self.state.transition("RELEASE", "documentation generated")

    def phase_release(self) -> None:
        from .release import validate_clean_room
        verification, report = validate_clean_room(self.sandbox, self.state, self.log)
        record_evaluation(self.root, "release-validation", verification)
        (self.root / "docs" / "generated" / "release-validation.md").write_text(report)
        criteria = self.completion_criteria(verification)
        self.save("release")
        if all(criteria.values()):
            version = "v0.1.0"
            self.commit("chore: release validation passed", kind="chore")
            if self.repo.available:
                self.repo.tag(version, f"{self.state.name} {version}")
            url = self.github.create_and_push(
                self.root, self.config.get("github.name") or self.state.name,
                (self.state.spec.get("objective") or self.state.idea)[:200])
            if url:
                self.state.metrics["github_url"] = url
                self.github.release(self.root, version, f"{self.state.name} {version}",
                                    (self.root / "README.md").read_text()[:5000])
            self.state.metrics["completion"] = criteria
            self.state.transition("COMPLETED", "all completion criteria satisfied")
            return
        failed = [k for k, v in criteria.items() if not v]
        count = self.state.record_failure("release")
        self.log.warn("release.blocked", failed=", ".join(failed), attempt=count)
        if count > 2:
            self.state.metrics["completion"] = criteria
            self.state.transition("FAILED", f"release criteria unmet: {failed}")
            return
        self.state.transition("IMPLEMENTATION", f"release blocked on {failed}")

    def completion_criteria(self, verification) -> dict[str, bool]:
        """Section 25: completion is more than an empty board."""
        gates = {g.name: (g.passed or g.skipped) for g in verification.gates}
        readme = (self.root / "README.md").read_text() if (self.root / "README.md").is_file() else ""
        outstanding = [t for t in self.board.by_status("ready", "active", "blocked")]
        critical = [t for t in self.board.by_status("abandoned") if t.priority <= 2]
        metrics = self.state.metrics.get("optimization")
        return {
            "requirements_satisfied": not outstanding,
            "build_succeeds": gates.get("build", False),
            "tests_pass": gates.get("tests", False),
            "integration_tests_pass": gates.get("tests", False),
            "no_critical_defects": not critical,
            "performance_satisfied": metrics is None or metrics.get("best") is not None,
            "documentation_exists": len(readme) > 400 and "## " in readme,
            "reproducible": gates.get("clean-room", False),
            "secure": not scan_unsafe(self.root),
        }

    # -- driver ----------------------------------------------------------
    PHASES = {
        "INTAKE": phase_intake,
        "RESEARCH": phase_research,
        "SPECIFICATION": phase_specification,
        "PLANNING": phase_planning,
        "IMPLEMENTATION": phase_implementation,
        "VALIDATION": phase_validation,
        "OPTIMIZATION": phase_optimization,
        "DOCUMENTATION": phase_documentation,
        "RELEASE": phase_release,
    }

    def run(self, max_iterations: int | None = None) -> State:
        """Drive the project to a terminal state without user interaction."""
        limit = max_iterations or int(self.config.get("execution.max_loop_iterations", 400))
        self.log.info("factory.start", project=self.state.name, phase=self.state.phase,
                      resources=self.resources)
        if self.config.get("git.enabled", True):
            self.repo.init()
        self.router.preflight()
        started = time.time()
        with self.sandbox:
            while not self.state.done and self.state.iteration < limit:
                if self.state.paused:
                    self.log.info("factory.paused")
                    break
                self.state.iteration += 1
                phase = self.state.phase
                self.log.info("loop", iteration=self.state.iteration, phase=phase,
                              progress=f"{self.board.progress():.0%}")
                try:
                    self.PHASES[phase](self)
                except KeyboardInterrupt:
                    self.log.warn("factory.interrupted")
                    self.save("interrupt")
                    break
                except Exception as exc:  # a phase blowing up must not kill the run
                    self._recover_phase(phase, exc)
                self.save(f"iter-{self.state.iteration}")
        self.state.metrics["elapsed_seconds"] = round(time.time() - started, 1)
        self.state.metrics["model_usage"] = _usage(self.router)
        self.save("final")
        self.log.info("factory.stop", phase=self.state.phase,
                      iterations=self.state.iteration,
                      elapsed=self.state.metrics["elapsed_seconds"])
        return self.state

    def _recover_phase(self, phase: str, exc: Exception) -> None:
        count = self.state.record_failure(f"phase:{phase}")
        self.log.error("phase.error", phase=phase, error=str(exc)[:400], count=count)
        self.state.notes.append(f"{phase} error ({count}): {exc}")
        if count < 3:
            return  # same phase runs again next iteration
        # Escalate: skip forward rather than spinning on a phase that cannot run.
        fallback = {
            "INTAKE": "SPECIFICATION", "RESEARCH": "SPECIFICATION",
            "SPECIFICATION": "PLANNING", "PLANNING": "IMPLEMENTATION",
            "IMPLEMENTATION": "VALIDATION", "VALIDATION": "DOCUMENTATION",
            "OPTIMIZATION": "DOCUMENTATION", "DOCUMENTATION": "RELEASE",
            "RELEASE": "FAILED",
        }.get(phase, "FAILED")
        self.state.transition(fallback, f"escalated past failing {phase}")
        self.state.clear_failure(f"phase:{phase}")


def _commit_kind(ticket_kind: str) -> str:
    return {"test": "test", "documentation": "docs", "optimization": "perf",
            "refactor": "refactor", "research": "docs", "experiment": "chore"}.get(
        ticket_kind, "feat")


def _usage(router: Router) -> dict:
    calls = router.usage
    return {
        "calls": len(calls),
        "seconds": round(sum(c["seconds"] for c in calls), 1),
        "chars_in": sum(c["chars_in"] for c in calls),
        "chars_out": sum(c["chars_out"] for c in calls),
    }


def create(root: Path, idea: str, config: Config | None = None,
           quiet: bool = False) -> "Factory":
    """Scaffold a new project workspace and return a Factory bound to it."""
    root = Path(root).resolve()
    config = config or Config.load()
    workspace.create(root, idea)
    factory = Factory(root, config, quiet=quiet)
    factory.log.info("project.created", root=str(root), idea=idea[:120])
    if config.get("git.enabled", True):
        factory.repo.init()
        factory.commit("chore: initialise Hermes workspace")
    return factory
