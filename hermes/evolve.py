"""AlphaEvolve-style evolutionary optimisation.

Correctness gates performance: a candidate that fails the tests is discarded
before its benchmark score is even considered. Selection is deliberately
sceptical, because a benchmark is a measurement rather than a fact: an
improvement has to clear the noise the baseline itself exhibits, and it has to
survive being measured again. Parents are drawn from an archive rather than
from a single incumbent, so the search is not confined to the first hill it
happened to climb.
"""
from __future__ import annotations

import json
import math
import random
import re
import shutil
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .agents import Prompt, context_block
from .log import Logger, NULL
from .sandbox import Sandbox
from .verify import Verification, run_commands, scan_unsafe

#: Benchmarks must print this so the score is unambiguous.
SCORE_LINE = re.compile(r"^\s*(?:HERMES_)?SCORE\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                        re.MULTILINE)
_NUMBER = re.compile(r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")

#: Kinds of change the evolver can be asked for. The bandit chooses between
#: these, so they must be distinguishable strategies rather than synonyms.
FAMILIES = (
    "algorithm: replace the method with one of lower complexity",
    "data-structure: change how the data is laid out or indexed",
    "precompute: hoist repeated work out of the hot path, cache or memoise",
    "batching: amortise per-item overhead over larger units of work",
    "allocation: reduce copying, allocation and indirection",
    "parallelism: use concurrency or vectorisation already available here",
    "io: change how data is read, written or buffered",
    "micro: tighten the inner loop without changing the approach",
)


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
    #: Which mutation family produced it, for the bandit's feedback.
    family: str = ""
    #: Every measurement taken of this candidate, not just the one that counted.
    samples: list[float] = field(default_factory=list)
    seconds: float = 0.0


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

    def aggregate(self, samples: list[float]) -> float | None:
        """The one number that represents a set of repeated measurements.

        The median, because benchmark noise on a shared machine is one-sided:
        a run can be arbitrarily slowed by something else on the box, but it
        cannot run faster than the hardware allows, so a mean is dragged
        around by outliers that say nothing about the code.
        """
        return statistics.median(samples) if samples else None


def parse_score(output: str) -> float | None:
    """Prefer an explicit ``SCORE: <n>`` line; fall back to the last number."""
    matches = SCORE_LINE.findall(output or "")
    if matches:
        return float(matches[-1])
    numbers = _NUMBER.findall((output or "").strip().splitlines()[-1] if output.strip() else "")
    return float(numbers[-1]) if numbers else None


class Bandit:
    """UCB1 over mutation families.

    Which kind of change pays off is a property of the project, not something
    knowable in advance: a tight numeric kernel rewards vectorisation where an
    IO-bound pipeline rewards buffering. Trying families uniformly spends most
    of the budget on the ones this project has already shown do not work, and
    always taking the current best never discovers the one that does. UCB1 is
    the standard answer to exactly that trade, and its exploration term decays
    on its own as evidence accumulates.
    """

    def __init__(self, arms: tuple[str, ...] | list[str]):
        self.counts: dict[str, int] = {arm: 0 for arm in arms}
        self.rewards: dict[str, float] = {arm: 0.0 for arm in arms}

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def value(self, arm: str) -> float:
        count = self.counts[arm]
        if not count:
            return math.inf  # an untried family is worth one trial on its own
        mean = self.rewards[arm] / count
        return mean + math.sqrt(2 * math.log(max(self.total, 1)) / count)

    def select(self) -> str:
        return max(self.counts, key=self.value)

    def update(self, arm: str, reward: float) -> None:
        if arm not in self.counts:
            return
        self.counts[arm] += 1
        self.rewards[arm] += reward

    def report(self) -> dict[str, dict[str, float]]:
        return {
            arm: {"tried": self.counts[arm],
                  "mean_reward": round(self.rewards[arm] / self.counts[arm], 3)
                  if self.counts[arm] else 0.0}
            for arm in self.counts
        }


class Evolution:
    """Drives generations of candidates against a benchmark."""

    def __init__(self, sandbox: Sandbox, agents, state, objective: Objective,
                 targets: list[str], cfg: dict, log: Logger = NULL, governor=None):
        #: Optional :class:`~hermes.resources.Governor`. Selection here is
        #: decided by timings, so a benchmark that shares the machine with a
        #: model call measures the contention instead of the candidate.
        self.governor = governor
        self.sandbox = sandbox
        self.agents = agents
        self.state = state
        self.objective = objective
        self.targets = targets
        self.cfg = cfg
        self.log = log
        self.history: list[Candidate] = []
        self.best: Candidate | None = None
        #: Spread of repeated measurements of unchanged code: the floor below
        #: which a "gain" is indistinguishable from the machine's own wobble.
        self.noise = 0.0
        #: Wall time of the incumbent's benchmark, used to screen slow runs.
        self.best_seconds = 0.0
        self.last_seconds = 0.0
        self.bandit = Bandit(FAMILIES)
        #: Two-sided memory. Only feeding back rejections tells the evolver
        #: what not to do and nothing about what this project rewards.
        self.helped: list[str] = []
        self.hurt: list[str] = []
        # Seeded: two runs of the same project should make the same choices.
        self.rng = random.Random(20250915)
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
            "noise": self.noise,
            "best": asdict(self.best) if self.best else None,
            "history": [asdict(c) for c in self.history],
            "pareto": [asdict(c) for c in self.pareto()],
            "archive": [c.generation for c in self.archive()],
            "families": self.bandit.report(),
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
    def measure(self, timeout: int | None = None) -> tuple[bool, float | None, Verification]:
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
        with self._quiet():
            result = self.sandbox.run(self.objective.command,
                                      timeout=timeout or self.full_timeout())
        verification.results.append(result)
        self.last_seconds = result.seconds
        if not result.ok:
            from .verify import Gate
            verification.gates.append(Gate("benchmark", False, result.summary(1200)))
            return True, None, verification
        return True, parse_score(result.stdout + "\n" + result.stderr), verification

    def full_timeout(self) -> int:
        return int(self.cfg.get("benchmark_timeout", 900))

    def screen_timeout(self) -> int:
        """Bound a candidate's first run by what the incumbent already achieves.

        Successive halving, in the only form a generated project supports:
        there is no cheap variant of the benchmark to screen on, but there is
        a time budget, and most losing candidates lose by being slow. Cutting
        them off at a multiple of the incumbent's time turns the worst
        generations from a full timeout into a few extra seconds. Raise
        ``screen_factor`` to switch it off; the cost of it being wrong is one
        discarded candidate, and the log says which.
        """
        factor = float(self.cfg.get("screen_factor", 2.0))
        if not self.best_seconds or factor <= 0:
            return self.full_timeout()
        return max(30, min(self.full_timeout(), int(factor * self.best_seconds) + 5))

    def _quiet(self):
        """Hold the machine as still as it can be made for a measurement."""
        return self.governor.measuring() if self.governor else nullcontext()

    def repeat(self, times: int, timeout: int | None = None) -> list[float]:
        """Re-run the benchmark on the code as it stands right now."""
        samples = []
        with self._quiet():
            for _ in range(max(0, times)):
                result = self.sandbox.run(self.objective.command,
                                          timeout=timeout or self.full_timeout())
                score = parse_score(result.stdout + "\n" + result.stderr) if result.ok else None
                if score is not None:
                    samples.append(score)
                    self.best_seconds = max(self.best_seconds, result.seconds)
        return samples

    # -- selection -------------------------------------------------------
    def margin(self, incumbent: float | None) -> float:
        """How much better a candidate must be before the gain is believed.

        Two floors, and a gain has to clear both: the measurement noise of
        the baseline, and a relative gain small enough not to be worth a
        regression risk. Without the first, the search promotes timing wobble
        to incumbent and then spends its whole budget failing to beat it.
        """
        relative = abs(incumbent) * float(self.cfg.get("min_relative_gain", 0.01)) \
            if incumbent is not None else 0.0
        return max(self.noise, relative)

    def significant(self, score: float | None, incumbent: float | None) -> bool:
        if not self.objective.better(score, incumbent):
            return False
        if incumbent is None:
            return True
        return abs(score - incumbent) >= self.margin(incumbent)

    def archive(self) -> list[Candidate]:
        """Correct candidates worth mutating again.

        The top of the ranking plus the Pareto front: the best score is the
        obvious parent, but a candidate that is slightly worse and much
        smaller is often the one a further change can actually build on.
        """
        limit = int(self.cfg.get("archive", 12))
        correct = [c for c in self.history if c.correct and c.score is not None]
        ranked = sorted(correct, key=lambda c: c.score,
                        reverse=self.objective.direction == "maximize")
        keep = ranked[:limit]
        for candidate in self.pareto():
            if candidate not in keep:
                keep.append(candidate)
        return keep

    def sample_parent(self) -> Candidate:
        """Choose what to mutate: usually the incumbent, sometimes not.

        Always mutating the best candidate is 1+1 hill climbing, which cannot
        leave a local optimum: once the obvious change is in, every further
        proposal is judged against it and the search stalls. Occasionally
        starting from a worse but different candidate is what makes a second
        basin reachable at all.
        """
        pool = [c for c in self.archive() if self.snapshot_exists(c.generation)]
        if len(pool) <= 1 or self.rng.random() < 0.7:
            return self.best
        return self.rng.choice(pool)

    def snapshot_exists(self, generation: int) -> bool:
        return (self.checkpoints / f"gen-{generation:03d}").is_dir()

    def baseline(self) -> Candidate:
        correct, score, verification = self.measure()
        candidate = Candidate(0, None, "baseline", score, correct,
                              list(self.targets), family="baseline",
                              notes="" if correct else verification.failure_output(1500))
        if correct and score is not None:
            self.best_seconds = self.last_seconds
            # Measure the machine, not just the code: the spread of the same
            # benchmark on unchanged code is the smallest gain that can mean
            # anything, and everything downstream is compared against it.
            samples = [score] + self.repeat(int(self.cfg.get("baseline_repeats", 3)) - 1)
            candidate.samples = samples
            candidate.score = self.objective.aggregate(samples)
            self.noise = statistics.pstdev(samples) if len(samples) > 1 else 0.0
            candidate.seconds = self.best_seconds
            self.best = candidate
            self.snapshot(0)
        self.history.append(candidate)
        self.log.info("evolve.baseline", correct=correct, score=candidate.score,
                      noise=round(self.noise, 6), samples=len(candidate.samples))
        self.save()
        return candidate

    # -- candidate generation --------------------------------------------
    def propose(self, generation: int, family: str, parent: Candidate) -> dict:
        evolver = self.agents["evolver"]
        prompt = Prompt(evolver, log=self.log)
        prompt.add("context", context_block(self.state, ", ".join(self.targets)),
                   priority=30, floor=600)
        prompt.add("objective",
                   f"OBJECTIVE: {self.objective.direction} {self.objective.metric}\n"
                   f"MEASURED BY: {self.objective.command}\n"
                   f"CURRENT BEST SCORE: {self.best.score if self.best else None}\n"
                   f"SCORE OF THE CODE BELOW: {parent.score}\n"
                   f"SMALLEST MEANINGFUL GAIN: {self.margin(parent.score):.6g} "
                   f"(anything less is measurement noise)\n"
                   f"GENERATION: {generation}", required=True)
        code = prompt.add("code", priority=10, floor=2000)
        prompt.add("history",
                   (("CHANGES THAT HELPED HERE:\n"
                     + "\n".join(f"- {h}" for h in self.helped[-6:]) + "\n\n")
                    if self.helped else "")
                   + (("CHANGES THAT DID NOT HELP HERE:\n"
                       + "\n".join(f"- {r}" for r in self.hurt[-12:]))
                      if self.hurt else ""), priority=20, floor=400)
        prompt.add("direction", f"THIS GENERATION'S DIRECTION: {family}\n\n"
                   "Propose ONE focused change in that direction that improves the "
                   "objective while keeping every test passing. Do not repeat a change "
                   "listed as not helping. Return JSON:\n"
                   '{"mutation": "one line naming the change", '
                   '"mechanism": "why this is faster/better, mechanically", '
                   '"files": [{"path": "", "content": "COMPLETE new file contents"}], '
                   '"risk": "what could break"}', required=True)
        # The candidate must return complete files, so it has to be shown
        # complete files: the code block takes what the fixed blocks leave,
        # shared between the targets, and a target that does not fit is not
        # offered at all rather than offered as a fragment.
        code.fill("CURRENT CODE:\n" + self._sources(max(0, code.room() - 14)))
        return evolver.ask_json(prompt.render())

    def _sources(self, budget: int) -> str:
        """The evolution targets, whole, within *budget* characters."""
        targets = self.targets[:4]
        share = budget // max(1, len(targets))
        sources = []
        for target in targets:
            text = self.sandbox.read(target, limit=share)
            if text:
                sources.append(f"--- {target} ---\n{text}")
        return "\n\n".join(sources)

    def remember(self, family: str, mutation: str, improved: bool,
                 detail: str = "") -> None:
        record = f"{mutation}{f' ({detail})' if detail else ''}"
        (self.helped if improved else self.hurt).append(record)

    # -- the loop ---------------------------------------------------------
    def run(self, max_generations: int = 20, patience: int = 5) -> Candidate | None:
        if self.best is None:
            base = self.baseline()
            if not base.correct or base.score is None:
                self.log.warn("evolve.no_baseline",
                              reason="baseline is not correct or produces no score")
                return None
        population = max(1, int(self.cfg.get("population", 3)))
        stagnant = 0

        for generation in range(1, max_generations + 1):
            improved_this_round = False
            for _ in range(population):
                improved_this_round |= self._attempt(len(self.history))
                self.save()
                if self.objective.reached(self.best.score):
                    self.log.info("evolve.target_reached", score=self.best.score)
                    self.restore(self.best.generation)
                    self.save()
                    return self.best
            stagnant = 0 if improved_this_round else stagnant + 1
            if stagnant >= patience:
                self.log.info("evolve.plateau", generations=generation)
                break

        if self.best:
            self.restore(self.best.generation)
        self.save()
        return self.best

    def _attempt(self, index: int) -> bool:
        """One candidate: propose, apply, measure, select. True when promoted."""
        parent = self.sample_parent()
        self.restore(parent.generation)
        family = self.bandit.select()
        try:
            proposal = self.propose(index, family, parent)
        except Exception as exc:  # noqa: BLE001 - a bad proposal is just a bad candidate
            self.log.warn("evolve.propose_failed", index=index, error=str(exc)[:200])
            return False

        mutation = str(proposal.get("mutation", "unnamed"))[:200]
        written = []
        for item in proposal.get("files") or []:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).lstrip("/")
            if path and isinstance(item.get("content"), str) and not path.startswith(
                    (".hermes/", ".git/")):
                self.sandbox.write(path, item["content"])
                written.append(path)
        if not written:
            self.bandit.update(family, 0.0)
            return False

        correct, score, verification = self.measure(timeout=self.screen_timeout())
        candidate = Candidate(index, parent.generation, mutation, score, correct,
                              written, family=family,
                              samples=[score] if score is not None else [],
                              seconds=self.last_seconds,
                              metrics={"mechanism": proposal.get("mechanism", "")})
        self.history.append(candidate)

        if not correct:
            candidate.notes = "failed correctness: " + verification.failure_output(800)
            self.remember(family, mutation, False, "broke tests")
            self.bandit.update(family, 0.0)
            self.log.info("evolve.discard", index=index, reason="incorrect")
            self.restore(parent.generation)
            return False

        self.snapshot(index)  # any correct candidate may become a parent later
        if not self.significant(score, self.best.score):
            candidate.notes = f"no significant gain ({score} vs {self.best.score})"
            self.remember(family, mutation, False, f"score {score}")
            # Correct but not better is still worth more than broken: it can
            # be mutated again, so the family keeps a fraction of the credit.
            self.bandit.update(family, 0.1 if score is not None else 0.0)
            self.log.info("evolve.discard", index=index, score=score,
                          best=self.best.score, margin=round(self.margin(self.best.score), 6))
            self.restore(parent.generation)
            return False

        # It looks better. Measure it again before believing it: the first
        # sample is exactly the one selection is biased towards over-trusting.
        samples = [score] + self.repeat(int(self.cfg.get("baseline_repeats", 3)) - 1)
        candidate.samples = samples
        candidate.score = self.objective.aggregate(samples)
        if not self.significant(candidate.score, self.best.score):
            candidate.notes = (f"gain did not reproduce ({score} once, "
                               f"{candidate.score} over {len(samples)} runs)")
            self.remember(family, mutation, False, "gain did not reproduce")
            self.bandit.update(family, 0.1)
            self.log.info("evolve.unconfirmed", index=index, first=score,
                          confirmed=candidate.score, best=self.best.score)
            self.restore(parent.generation)
            return False

        self.snapshot(index)  # re-taken: the confirmation runs may have written
        delta = f"{self.best.score} -> {candidate.score}"
        self.best = candidate
        self.best_seconds = candidate.seconds or self.best_seconds
        self.remember(family, mutation, True, f"score {candidate.score}")
        self.bandit.update(family, 1.0)
        self.log.info("evolve.improved", index=index, delta=delta, family=family[:20],
                      mutation=mutation[:80])
        return True

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
        lines = [f"objective: {self.objective.direction} {self.objective.metric}",
                 f"measurement noise: {self.noise:.6g}"]
        for candidate in self.history:
            mark = "best" if self.best and candidate.generation == self.best.generation else (
                "ok" if candidate.correct else "bad")
            lines.append(f"gen {candidate.generation:>3} [{mark:>4}] "
                         f"score={candidate.score} "
                         f"[{candidate.family.split(':')[0]}] {candidate.mutation[:50]}")
        lines.append("\nmutation families:")
        for family, stats in self.bandit.report().items():
            if stats["tried"]:
                lines.append(f"  {family.split(':')[0]:<14} tried={stats['tried']} "
                             f"mean reward={stats['mean_reward']}")
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
