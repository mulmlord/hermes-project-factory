"""The autonomous factory loop.

One object owns the whole lifecycle: intake, research, specification,
planning, implementation, validation, optimisation, documentation, release.
It never asks the user anything; every branch resolves to a decision.
"""
from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from . import workspace
from .agents import Prompt, build as build_agents, context_block
from .config import Config
from .evolve import Evolution, objective_from_spec
from .git_ops import GitHub, Issues, Repo
from .implement import Change, execute_ticket, generate
from .journal import Journal
from .log import Logger, current_scope, for_workspace, scope
from .models import Router
from .openhands import OpenHands
from .planning import architect, decompose, generate_tickets, intake
from .research import investigate, topics_for
from .resources import Governor
from .sandbox import Sandbox
from .server import running as server_running
from .sources import STORE as SOURCE_STORE, Librarian
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
        self.journal = Journal(self.root, self.log)
        self.governor = Governor(self.config.get("resources", {}), self.log)
        self.router = Router(self.config, self.log, self.governor)
        self.agents = build_agents(self.router, self.log)
        self.sandbox = Sandbox(self.root, self.config.get("sandbox", {}), self.log)
        self.repo = Repo(self.root, self.log, self.config.get("git.author"))
        self.github = GitHub(self.config.get("github", {}), self.log)
        self.issues = Issues(self.github, self.root, self.log,
                             int(self.config.get("github.issue_batch", 20)))
        self.openhands = OpenHands(self.config.get("openhands", {}),
                                   self.config.model_for("coding"), self.log)
        self.resources = self.governor.snapshot()
        self._pool: ThreadPoolExecutor | None = None
        self._prefetch: tuple[str, Future] | None = None

    # -- helpers ---------------------------------------------------------
    def tree(self) -> str:
        return workspace.tree(self.root)

    def context(self, extra: str = "") -> str:
        return context_block(self.state, self.tree(), extra)

    def save(self, label: str = "") -> None:
        self.state.save(self.root)
        self.board.save()
        workspace.write_context(self.root, self.state, self._board_extra())
        if self.config.get("github.issues", True):
            # Free while GitHub is disabled or the project has no repository,
            # and free again on a board that did not move.
            self.issues.sync(self.board.tickets.values())
        if label and self.state.iteration % max(
                1, int(self.config.get("execution.checkpoint_every", 1))) == 0:
            checkpoint(self.root, label)

    def _board_extra(self) -> str:
        return f"## Board\n```\n{self.board.summary()}\n```\n"

    def commit(self, message: str, kind: str = "chore") -> str | None:
        """Returns the short SHA so the journal can point at the real commit."""
        if not self.config.get("git.enabled", True):
            return None
        return self.repo.commit(message, kind=kind)

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
        text = "\n".join(lines)
        return text if len(text) <= limit else self._compress(text, limit)

    def _compress(self, text: str, limit: int) -> str:
        """Summarise overlong material instead of truncating it.

        Cutting at a character count silently drops whichever findings sort
        last, which is how the constraint the architect needed disappears.
        This is the one job in the factory that wants the small fast model,
        and it is what `routing.simple` is for.
        """
        summariser = self.agents["summariser"]
        prompt = Prompt(summariser, log=self.log)
        findings_block = prompt.add("findings", priority=10, floor=2000)
        prompt.add("instructions",
                   f"Compress this to at most {limit // 12} words for an architect who "
                   "must choose a design from it. Keep every decision, constraint, "
                   "measured number, named technology and rejected alternative. "
                   'Return JSON: {"findings": ["one finding per entry"]}', required=True)
        # More findings than one pass can hold is a real possibility late in a
        # long project; the oldest are the ones already folded into the spec,
        # so what is kept here is the tail.
        head = f"RESEARCH FINDINGS ({len(text)} characters):\n"
        keep = max(0, findings_block.room() - len(head))
        findings_block.fill(head + (text[-keep:] if keep else ""))
        try:
            payload = summariser.ask_json(prompt.render())
            findings = [f"- {item}" for item in (payload.get("findings") or []) if item]
        except Exception as exc:  # noqa: BLE001 - truncation is a valid fallback
            self.log.warn("research.compress_failed", error=str(exc)[:200])
            return text[:limit]
        return "\n".join(findings)[:limit] or text[:limit]

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
                log=self.log, librarian=self._librarian())
            self._record_decisions(payload.get("decisions", []))
            self._queue_research_tickets(payload.get("tickets", []))
        self.save("research")
        self.commit("docs: record research findings and decisions", kind="docs")
        self.state.transition("SPECIFICATION", f"{len(topics)} research topic(s)")

    def _librarian(self) -> Librarian | None:
        """A fetcher, only where the operator enabled a network budget.

        One per round, so the page budget is per round rather than per run,
        and None by default: research reaches the network only on purpose.
        """
        if not self.config.get("research.network", False):
            return None
        self._record_evidence_policy()
        return Librarian(self.sandbox, self.root, self.config.get("research", {}),
                         self.log)

    def _record_evidence_policy(self) -> None:
        """Say in the project what becomes of the pages it retrieves.

        Written once, and only for a project that actually reaches the
        network, so a project that never fetched is not handed a decision it
        never took.
        """
        directory = self.root / "docs" / "decisions"
        if any(directory.glob("*-retrieved-research-pages*.md")):
            return
        workspace.write_decision(
            self.root, workspace.next_decision_number(self.root),
            "Retrieved research pages are kept as evidence, not published",
            "Networked research stores each page it reads under "
            f"`{SOURCE_STORE}/` so that a citation can be re-read rather than "
            "taken on trust. Those bytes are third-party material, and up to "
            "several hundred kilobytes of it per research round.",
            "Keep the pages in the working tree, and keep them out of the "
            "repository. The directory is gitignored, so it is never committed "
            "and never pushed, and the clean-room export drops it even where an "
            "older project committed it. The extracted text and the citation "
            "stay in `docs/research/`, which is published.",
            "Evidence survives for whoever audits a finding on this machine. A "
            "fresh clone has the citations but not the copies, so a claim is "
            "re-checked against the source itself rather than against a stale "
            "snapshot of it.",
            "Publishing the copies would republish someone else's pages under "
            "this project's licence. Keeping only the extracted text would make "
            "a citation unverifiable the moment the page changed.")

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
        self._publish_board()
        self.state.transition("IMPLEMENTATION", f"{len(self.board.tickets)} tickets")

    def _publish_board(self) -> None:
        """Create the repository as soon as there is a board worth watching.

        The issue projection can only act once a remote exists, and the remote
        used to arrive in phase_release -- by which point every ticket is
        already closed and there is nothing left to observe. So the repository
        is created here, where the board first exists, and the code is held
        back until a ticket has passed its gates: repositories are private by
        default, and a scaffold nobody has validated is not worth showing.
        """
        if not (self.config.get("github.enabled", False)
                and self.config.get("github.early", True)):
            return
        url = self.github.create(
            self.root, self.config.get("github.name") or self.state.name,
            (self.state.spec.get("objective") or self.state.idea)[:200])
        if url:
            self.state.metrics["github_url"] = url

    def _push_first_green(self) -> None:
        """Push the code once something has actually passed, and only once."""
        if not self.config.get("github.enabled", False):
            return
        if self.state.metrics.get("github_pushed"):
            return
        if not self.state.metrics.get("github_url"):
            return
        if self.github.push(self.root):
            self.state.metrics["github_pushed"] = True

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

        # Everything below belongs to this ticket, down to the model call that
        # generated a patch and the command that rejected it.
        with scope(ticket=ticket.id):
            if self.openhands.should_handle(ticket):
                handled = self._run_openhands(ticket)
                if handled:
                    return

            outcome = execute_ticket(
                self.agents, self.sandbox, self.state, self.board, ticket, self.tree,
                journal=self.journal,
                max_attempts=int(self.config.get("execution.max_ticket_attempts", 3)),
                candidates=int(self.config.get("execution.candidates", 1)),
                allow_edits=bool(self.config.get("execution.edit_protocol", True)),
                prefetched=self._take_prefetch(ticket),
                on_apply=lambda: self._prefetch_next(ticket),
                log=self.log)

            if outcome["status"] == "completed":
                self.board.move(ticket.id, "completed", "all gates passed")
                self.state.clear_failure(ticket.id)
                record_evaluation(self.root, ticket.id, outcome["verification"],
                                  {"review": outcome.get("review", {})})
                sha = self.commit(f"{_commit_kind(ticket.kind)}: {ticket.title}\n\n"
                                  f"Ticket {ticket.id}. {outcome['change'].summary}",
                                  kind=_commit_kind(ticket.kind))
                self.journal.record(ticket, outcome, commit=sha or "")
                self.log.info("ticket.completed",
                              progress=f"{self.board.progress():.0%}")
                self._push_first_green()
            else:
                self._handle_ticket_failure(ticket, outcome)
            self.save(f"ticket-{ticket.id}")

    # -- pipelining ------------------------------------------------------
    def _prefetch_next(self, current) -> None:
        """Generate the next ticket's code while this one is being verified.

        Verification is minutes of CPU during which the GPU sits idle, and
        the two pieces of hardware are not in contention. The prefetched work
        is strictly read-only -- one model call plus file reads, never a
        write -- so the worst case is a payload built against files the
        current ticket then changed. That payload fails its gates and the
        normal retry path regenerates it, which is the same cost as not
        having prefetched at all.
        """
        if not self.config.get("execution.prefetch", True) or self._prefetch:
            return
        candidate = self._independent_ticket(current)
        if candidate is None:
            return
        if not self.governor.admit_inference():
            # Sequential generation is the fallback, and it is exactly what the
            # loop did before prefetching existed. Nothing is lost but overlap.
            self.log.info("prefetch.deferred", ticket=candidate.id,
                          reason=self.governor.last_refusal)
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hermes-pf")
        self.log.info("prefetch.start", ticket=candidate.id, during=current.id)
        # The pool thread starts with an empty context, so the run is carried
        # across by hand. What it does belongs to the candidate, not to the
        # ticket whose gates are running, and to no attempt of it yet.
        inherited = dict(current_scope(), ticket=candidate.id,
                         attempt=None, prefetch=True)
        # Read here, not in the worker: the journal and the tree belong to the
        # loop thread, which goes on writing to them while this runs.
        args = (self.agents, self.sandbox, self.state, candidate, self.tree())
        kwargs = dict(prior=self.journal.block(candidate, self.board),
                      hints=self.journal.suggested_files(candidate, self.board),
                      allow_edits=bool(self.config.get("execution.edit_protocol", True)),
                      log=self.log)

        def generate_ahead() -> dict:
            with scope(**inherited):
                return generate(*args, **kwargs)

        self._prefetch = (candidate.id, self._pool.submit(generate_ahead))

    def _independent_ticket(self, current):
        """The ticket that will probably run next, unless it could collide.

        Anything still depending on the running ticket is blocked rather than
        ready, so the only remaining overlap is two tickets writing the same
        file; that pair is left alone and generated in sequence.
        """
        candidate = self.board.next_ticket()
        if candidate is None or candidate.id == current.id:
            return None
        if current.id in candidate.dependencies:
            return None
        busy = {a.lstrip("/") for a in current.artifacts}
        if busy & {a.lstrip("/") for a in candidate.artifacts}:
            return None
        return candidate

    def _take_prefetch(self, ticket) -> dict | None:
        """The prefetched payload, when the ticket it was made for came up."""
        if not self._prefetch:
            return None
        ticket_id, future = self._prefetch
        self._prefetch = None
        if ticket_id != ticket.id:
            future.cancel()  # a running call cannot be stopped; drop the result
            return None
        try:
            payload = future.result()
        except Exception as exc:  # noqa: BLE001 - fall back to generating it now
            self.log.warn("prefetch.failed", ticket=ticket.id, error=str(exc)[:200])
            return None
        self.log.info("prefetch.hit", ticket=ticket.id)
        return payload if isinstance(payload, dict) else None

    def _stop_prefetch(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        self._prefetch = None

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
            sha = self.commit(f"{_commit_kind(ticket.kind)}: {ticket.title}\n\n"
                              f"Ticket {ticket.id}, implemented via OpenHands.",
                              kind=_commit_kind(ticket.kind))
            # OpenHands edits the workspace itself, so the commit is the only
            # record of which files it actually touched.
            self.journal.record(
                ticket,
                {"status": "completed", "verification": verification,
                 "change": Change(summary="implemented by the OpenHands agent",
                                  written=self.repo.files_in())},
                commit=sha or "")
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
                                  log=self.log, librarian=self._librarian())
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
        # A recorded dead end is worth more to the next agent than a success:
        # it is the only thing stopping the same approach being tried again.
        self.journal.record(
            ticket, outcome,
            status="abandoned" if self.board.tickets[ticket.id].status == "abandoned"
            else "failed")

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
                           self.config.get("evolution", {}), self.log,
                           governor=self.governor)
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

    def run(self, max_iterations: int | None = None,
            deadline: float | None = None) -> State:
        """Drive the project to a terminal state without user interaction.

        *deadline* is an absolute wall-clock time after which the loop stops
        between iterations. Iterations bound how many phases run, not how long
        one takes, so without it a single stuck command is bounded only by the
        sandbox timeout. A phase already running still finishes: the real
        ceiling is the deadline plus one phase.
        """
        limit = max_iterations or int(self.config.get("execution.max_loop_iterations", 400))
        # One identifier for this run, on every record it produces and in the
        # state it leaves behind: a resumed project writes into the same log
        # file as the run before it, and only this separates them. The CLI has
        # usually minted one already; taking it keeps a single `hermes run`
        # from reporting two.
        run = current_scope().get("run") or uuid.uuid4().hex[:8]
        self.state.metrics["run"] = run
        with scope(run=run):
            self.log.info("factory.start", project=self.state.name, phase=self.state.phase,
                          resources=self.resources)
            if self.config.get("git.enabled", True):
                self.repo.init()
            # The inference server is needed for exactly as long as this call
            # runs, which is why it is started here instead of at boot. Leaving
            # the block releases it, so neither a preflight failure nor a crash
            # in the loop leaves a model sitting in VRAM.
            with server_running(self.config, self.router, self.log):
                self._loop(limit, deadline)
        return self.state

    def _loop(self, limit: int, deadline: float | None) -> None:
        self.router.preflight()
        started = time.time()
        with self.sandbox:
            while not self.state.done and self.state.iteration < limit:
                if deadline is not None and time.time() >= deadline:
                    self.log.warn("factory.deadline", phase=self.state.phase,
                                  iteration=self.state.iteration)
                    self.state.metrics["deadline_reached"] = True
                    self.state.notes.append("stopped: wall-clock deadline reached")
                    break
                if self.state.paused:
                    self.log.info("factory.paused")
                    break
                self.state.iteration += 1
                phase = self.state.phase
                with scope(phase=phase, iteration=self.state.iteration):
                    self.log.info("loop", progress=f"{self.board.progress():.0%}")
                    try:
                        self.PHASES[phase](self)
                    except KeyboardInterrupt:
                        self.log.warn("factory.interrupted")
                        self.save("interrupt")
                        break
                    except Exception as exc:  # a phase blowing up must not kill the run
                        self._recover_phase(phase, exc)
                    self.save(f"iter-{self.state.iteration}")
            self._stop_prefetch()
        self.state.metrics["elapsed_seconds"] = round(time.time() - started, 1)
        self.state.metrics["model_usage"] = _usage(self.router)
        self.state.metrics["sandbox_seconds"] = round(self.sandbox.spent, 1)
        self.save("final")
        self.log.info("factory.stop", phase=self.state.phase,
                      iterations=self.state.iteration,
                      elapsed=self.state.metrics["elapsed_seconds"])

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
