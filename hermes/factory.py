"""The autonomous factory loop.

:class:`Factory` holds a project: its state, the resources the work runs
through, the scheduling that decides what happens next, and the recovery that
keeps a failure from ending the run. The work itself lives in
:mod:`hermes.phases`, one function per phase, and the loop here drives them
until the project reaches a terminal state.

It never asks the user anything; every branch resolves to a decision.
"""
from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from . import prompts, workspace
from .agents import Prompt, build as build_agents, context_block
from .classify import Classifier
from .config import Config
from .contextdb import open_db
from .experiment import Lab
from .git_ops import GitHub, Issues, Repo
from .implement import Ladder, generate
from .journal import Journal
from .log import Logger, current_scope, for_workspace, scope
from .models import Router
from .openhands import OpenHands
from .phases import PHASES
from .planning import decompose
from .research import investigate
from .resources import Governor
from .sandbox import Sandbox
from .server import running as server_running
from .sources import STORE as SOURCE_STORE, Librarian
from .state import State, checkpoint
from .tickets import Board
from .tools import Toolbox


class Factory:
    def __init__(self, root: Path, config: Config | None = None,
                 log: Logger | None = None, quiet: bool = False):
        self.root = Path(root).resolve()
        self.config = config or Config.load()
        self.log = log or for_workspace(self.root, self.config.get("logging.level"), quiet)
        self.state = State.load(self.root)
        self.board = Board(self.root)
        #: Where agents report what they did and read what the others did.
        #: See :mod:`hermes.contextdb`.
        self.context_db = open_db(self.root, self.log)
        self.journal = Journal(self.root, self.log, db=self.context_db)
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
        self.ladder = Ladder.from_config(self.config)
        self.classifier = Classifier.from_config(self.config, self.log)
        self.resources = self.governor.snapshot()
        self._pool: ThreadPoolExecutor | None = None
        self._prefetch: tuple[str, Future] | None = None
        self._lib: Librarian | None = None

    # -- helpers ---------------------------------------------------------
    def tree(self) -> str:
        return workspace.tree(self.root)

    def _toolbox(self) -> Toolbox:
        """A workspace-lookup budget for one generation. See :mod:`hermes.tools`."""
        return Toolbox(self.sandbox,
                       budget=int(self.config.get("execution.tool_calls", 6)),
                       rounds=int(self.config.get("execution.tool_rounds", 3)),
                       db=self.context_db, log=self.log)

    def _lab(self) -> Lab:
        """Where hypotheses are measured. See :mod:`hermes.experiment`.

        Built per experiment rather than held: it carries no state between
        them, and the agents and sandbox it works through are attributes the
        caller may have replaced since the factory was constructed.
        """
        return Lab(self.sandbox, self.agents, self.state,
                   self.config.get("research", {}), self.log,
                   governor=self.governor)

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
        prompt.add("instructions", prompts.compress_findings(limit // 12),
                   required=True)
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

    # -- research resources ----------------------------------------------
    def _librarian(self) -> Librarian | None:
        """The one fetcher for this run, or None: the network is opt-in.

        One per run, not one per call. Research is entered once here and again
        by every ticket failure triaged as ``research``, so a librarian built
        per call made ``max_fetches`` a number that nothing ever added up: ten
        research escalations fetched ten times the configured pages. The
        per-round share inside it is what still bounds a single round.
        """
        if not self.config.get("research.network", False):
            return None
        if self._lib is None:
            self._record_evidence_policy()
            self._lib = Librarian(self.sandbox, self.root,
                                  self.config.get("research", {}), self.log)
        return self._lib

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

    # -- scheduling ------------------------------------------------------
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
                      # A prefetch is a first attempt, so it is asked the way
                      # the first rung asks; anything harder is wasted on work
                      # that has not failed yet.
                      files=self.ladder.rung(1).files,
                      toolbox=self._toolbox(),
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

    # -- failure recovery -------------------------------------------------
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

    # -- driver ----------------------------------------------------------
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
                        PHASES[phase](self)
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
