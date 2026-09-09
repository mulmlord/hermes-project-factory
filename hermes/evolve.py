"""AlphaEvolve-style evolutionary optimisation.

Correctness gates performance: a candidate that fails the tests is discarded
before its benchmark score is even considered. Every generation records its
parent, its mutation, its correctness result and its score, and the working
tree is always restored from the current best before the next candidate.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .agents import context_block
from .log import Logger, NULL
from .sandbox import Sandbox
from .verify import Verification, run_commands, scan_unsafe

#: Benchmarks must print this so the score is unambiguous.
SCORE_LINE = re.compile(r"^\s*(?:HERMES_)?SCORE\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                        re.MULTILINE)
_NUMBER = re.compile(r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


@dataclass
class Candidate:
    generation: int
    parent: int | None
    mutation: str
    score: float | None = None
    correct: bool = False
    files: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    notes: str = ""
    at: float = field(default_factory=time.time)


class Objective:
    """A single measurable objective and how to compare two scores."""

    def __init__(self, metric: str, direction: str = "minimize",
                 command: str = "", threshold: float | None = None):
        self.metric = metric
        self.direction = "maximize" if str(direction).lower().startswith("max") else "minimize"
        self.command = command
        self.threshold = threshold

    def better(self, candidate: float | None, incumbent: float | None) -> bool:
        if candidate is None:
            return False
        if incumbent is None:
            return True
        return candidate > incumbent if self.direction == "maximize" else candidate < incumbent

    def reached(self, score: float | None) -> bool:
        if score is None or self.threshold is None:
            return False
        return self.better(score, self.threshold) or score == self.threshold


def parse_score(output: str) -> float | None:
    """Prefer an explicit ``SCORE: <n>`` line; fall back to the last number."""
    matches = SCORE_LINE.findall(output or "")
    if matches:
        return float(matches[-1])
    numbers = _NUMBER.findall((output or "").strip().splitlines()[-1] if output.strip() else "")
    return float(numbers[-1]) if numbers else None


class Evolution:
    """Drives generations of candidates against a benchmark."""

    def __init__(self, sandbox: Sandbox, agents, state, objective: Objective,
                 targets: list[str], cfg: dict, log: Logger = NULL):
        self.sandbox = sandbox
        self.agents = agents
        self.state = state
        self.objective = objective
        self.targets = targets
        self.cfg = cfg
        self.log = log
        self.history: list[Candidate] = []
        self.best: Candidate | None = None
        self.root = sandbox.root
        self.store = self.root / ".hermes" / "state" / "evolution.json"
        self.checkpoints = self.root / "artifacts" / "evolution"
        self.checkpoints.mkdir(parents=True, exist_ok=True)

    # -- persistence -----------------------------------------------------
    def save(self) -> None:
        self.store.parent.mkdir(parents=True, exist_ok=True)
        self.store.write_text(json.dumps({
            "objective": {"metric": self.objective.metric,
                          "direction": self.objective.direction,
                          "command": self.objective.command},
            "targets": self.targets,
            "best": asdict(self.best) if self.best else None,
            "history": [asdict(c) for c in self.history],
            "pareto": [asdict(c) for c in self.pareto()],
        }, indent=2, default=str))

    def snapshot(self, generation: int) -> Path:
        """Copy the current target files so any generation can be restored."""
        directory = self.checkpoints / f"gen-{generation:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        for target in self.targets:
            source = self.root / target
            if source.is_file():
                dest = directory / target.replace("/", "__")
                shutil.copy2(source, dest)
        return directory

    def restore(self, generation: int) -> None:
        directory = self.checkpoints / f"gen-{generation:03d}"
        for target in self.targets:
            saved = directory / target.replace("/", "__")
            if saved.is_file():
                (self.root / target).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(saved, self.root / target)

    # -- measurement -----------------------------------------------------
    def measure(self) -> tuple[bool, float | None, Verification]:
        """Correctness first, then the benchmark. Never score broken code."""
        commands = self.state.plan.get("commands") or {}
        verification = Verification()
        run_commands(self.sandbox, [commands.get("build", "")], "build",
                     verification, log=self.log)
        run_commands(self.sandbox, [commands.get("test", "")], "tests",
                     verification, log=self.log)
        hits = scan_unsafe(self.root, self.targets)
        if hits:
            from .verify import Gate
            verification.gates.append(Gate("security", False, "; ".join(hits[:3])))
            return False, None, verification
        if not verification.passed:
            return False, None, verification
        result = self.sandbox.run(self.objective.command,
                                  timeout=int(self.cfg.get("benchmark_timeout", 900)))
        verification.results.append(result)
        if not result.ok:
            from .verify import Gate
            verification.gates.append(Gate("benchmark", False, result.summary(1200)))
            return True, None, verification
        return True, parse_score(result.stdout + "\n" + result.stderr), verification

    def baseline(self) -> Candidate:
        correct, score, verification = self.measure()
        candidate = Candidate(0, None, "baseline", score, correct,
                              list(self.targets),
                              notes="" if correct else verification.failure_output(1500))
        self.history.append(candidate)
        if correct and score is not None:
            self.best = candidate
            self.snapshot(0)
        self.log.info("evolve.baseline", correct=correct, score=score)
        self.save()
        return candidate

    # -- candidate generation --------------------------------------------
    def propose(self, generation: int, rejected: list[str]) -> dict:
        sources = []
        for target in self.targets[:4]:
            text = self.sandbox.read(target, limit=12000)
            if text:
                sources.append(f"--- {target} ---\n{text}")
        best_score = self.best.score if self.best else None
        return self.agents["evolver"].ask_json(
            f"{context_block(self.state, ', '.join(self.targets))}\n\n"
            f"OBJECTIVE: {self.objective.direction} {self.objective.metric}\n"
            f"MEASURED BY: {self.objective.command}\n"
            f"CURRENT BEST SCORE: {best_score}\n"
            f"GENERATION: {generation}\n\n"
            f"CURRENT CODE:\n" + "\n\n".join(sources) + "\n\n"
            + (("MUTATIONS ALREADY TRIED THAT DID NOT HELP:\n"
                + "\n".join(f"- {r}" for r in rejected[-12:]) + "\n\n") if rejected else "")
            + "Propose ONE focused change that improves the objective while "
            "keeping every test passing. Do not repeat a mutation listed above. "
            "Prefer better algorithms and data structures over micro-optimisation. "
            "Return JSON:\n"
            '{"mutation": "one line naming the change", '
            '"mechanism": "why this is faster/better, mechanically", '
            '"files": [{"path": "", "content": "COMPLETE new file contents"}], '
            '"risk": "what could break"}'
        )

    # -- the loop ---------------------------------------------------------
    def run(self, max_generations: int = 20, patience: int = 5) -> Candidate | None:
        if self.best is None:
            base = self.baseline()
            if not base.correct or base.score is None:
                self.log.warn("evolve.no_baseline",
                              reason="baseline is not correct or produces no score")
                return None
        stagnant, rejected = 0, []

        for generation in range(1, max_generations + 1):
            parent = self.best
            self.restore(parent.generation)
            try:
                proposal = self.propose(generation, rejected)
            except Exception as exc:  # noqa: BLE001 - a bad proposal is just a bad generation
                self.log.warn("evolve.propose_failed", generation=generation,
                              error=str(exc)[:200])
                stagnant += 1
                continue

            mutation = str(proposal.get("mutation", "unnamed"))[:200]
            written = []
            for item in proposal.get("files") or []:
                path = str(item.get("path", "")).lstrip("/")
                if path and isinstance(item.get("content"), str) and not path.startswith(
                        (".hermes/", ".git/")):
                    self.sandbox.write(path, item["content"])
                    written.append(path)
            if not written:
                stagnant += 1
                continue

            correct, score, verification = self.measure()
            candidate = Candidate(generation, parent.generation, mutation, score,
                                  correct, written,
                                  metrics={"mechanism": proposal.get("mechanism", "")})
            self.history.append(candidate)

            if not correct:
                candidate.notes = "failed correctness: " + verification.failure_output(800)
                rejected.append(f"{mutation} (broke tests)")
                self.log.info("evolve.discard", generation=generation, reason="incorrect")
                self.restore(parent.generation)
                stagnant += 1
            elif self.objective.better(score, parent.score):
                self.snapshot(generation)
                self.best = candidate
                delta = f"{parent.score} -> {score}"
                self.log.info("evolve.improved", generation=generation, delta=delta,
                              mutation=mutation[:80])
                stagnant = 0
            else:
                candidate.notes = f"no improvement ({score} vs {parent.score})"
                rejected.append(f"{mutation} (score {score})")
                self.log.info("evolve.discard", generation=generation, score=score,
                              best=parent.score)
                self.restore(parent.generation)
                stagnant += 1

            self.save()
            if self.objective.reached(self.best.score):
                self.log.info("evolve.target_reached", score=self.best.score)
                break
            if stagnant >= patience:
                self.log.info("evolve.plateau", generations=generation)
                break

        if self.best:
            self.restore(self.best.generation)
        self.save()
        return self.best

    def pareto(self) -> list[Candidate]:
        """Correct candidates not dominated on (score, code size).

        Multi-objective tracking: a slower but far smaller candidate is kept
        so a later generation can build from it.
        """
        points = []
        for candidate in self.history:
            if not candidate.correct or candidate.score is None:
                continue
            size = sum(len(self.sandbox.read(f) or "") for f in candidate.files) or 1
            points.append((candidate, candidate.score, size))
        front = []
        for candidate, score, size in points:
            dominated = any(
                self.objective.better(other_score, score) and other_size <= size
                for other, other_score, other_size in points if other is not candidate
            )
            if not dominated:
                front.append(candidate)
        return front

    def report(self) -> str:
        lines = [f"objective: {self.objective.direction} {self.objective.metric}"]
        for candidate in self.history:
            mark = "best" if self.best and candidate.generation == self.best.generation else (
                "ok" if candidate.correct else "bad")
            lines.append(f"gen {candidate.generation:>3} [{mark:>4}] "
                         f"score={candidate.score} {candidate.mutation[:60]}")
        return "\n".join(lines)


def objective_from_spec(spec: dict, plan: dict) -> Objective | None:
    """Build an objective from the spec, or None when nothing is measurable."""
    raw = spec.get("optimization_objective") or {}
    metric = (raw.get("metric") or "").strip() if isinstance(raw, dict) else str(raw)
    command = (plan.get("commands") or {}).get("benchmark", "")
    if not metric or not command:
        return None
    direction = raw.get("direction", "minimize") if isinstance(raw, dict) else "minimize"
    return Objective(metric, direction, command)
