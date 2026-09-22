"""AlphaEvolve-style evolutionary optimisation.

Correctness gates performance: a candidate that fails the tests is discarded
before its benchmark score is even considered. Selection is deliberately
sceptical, because a benchmark is a measurement rather than a fact: an
improvement has to clear the noise the baseline itself exhibits, and it has to
survive being measured again. Parents are drawn from an archive rather than
from a single incumbent, so the search is not confined to the first hill it
happened to climb, and the archive can be split into islands that take the
generations in turn, so a second approach is not judged against the first
one's score before it has been given a generation to become good.
"""
from __future__ import annotations

import difflib
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

from . import prompts
from .agents import Prompt, context_block
from .implement import Change, apply_change
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
    #: The island it was born on, which is the archive it competes inside.
    island: int = 0
    #: Every measurement taken of this candidate, not just the one that counted.
    samples: list[float] = field(default_factory=list)
    seconds: float = 0.0


@dataclass(frozen=True)
class Tried:
    """A change that has already been paid for, and what it got.

    *changes* is the set of lines the candidate added to and removed from its
    parent, which is what a later proposal is compared against; *candidate* is
    kept whole rather than copied from, so the outcome read back to the
    evolver is the one selection finally recorded, not the one the branch that
    stored it happened to know.
    """
    parent: int
    changes: frozenset[str]
    candidate: Candidate

    def outcome(self) -> str:
        return self.candidate.notes or f"scored {self.candidate.score}"


class NoTargetFits(RuntimeError):
    """No evolution target can be shown whole inside the evolver's context.

    Raised rather than half-showing one: an anchor copied from a fragment is
    still exact, but a candidate cannot reason about a hot loop it was given
    the first half of, and the tail it never saw is where the cost usually is.
    """


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


def _significant(text: str) -> list[str]:
    """The lines of *text* that could change what a program does."""
    return [line.rstrip() for line in text.splitlines() if line.strip()]


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
        #: Separate archives, each with its own incumbent, taking the
        #: generations in turn. One island is the single population.
        self.islands = max(1, int(self.cfg.get("islands", 1)))
        self.migrate_every = int(self.cfg.get("migrate_every", 4))
        #: Which generations each island may mutate. Membership rather than
        #: ownership: migration lends a candidate to a neighbour without
        #: taking it from the island that earned it.
        self.members: list[set[int]] = [set() for _ in range(self.islands)]
        #: What each island promotes against. An island compared against the
        #: global best would be an island in name only.
        self.leaders: list[Candidate | None] = [None] * self.islands
        self.migrations = 0
        #: Every change that has been through the gates, by the parent it was
        #: made to, so the same one is not bought twice.
        self.tried: list[Tried] = []
        self.repeats = 0
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
            "repeats_refused": self.repeats,
            "islands": [{"leader": leader.generation if leader else None,
                         "score": leader.score if leader else None,
                         "generations": sorted(members)}
                        for leader, members in zip(self.leaders, self.members)],
            "migrations": self.migrations,
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

    def screen_timeout(self, seconds: float = 0.0) -> int:
        """Bound a candidate's first run by what its incumbent already achieves.

        Successive halving, in the only form a generated project supports:
        there is no cheap variant of the benchmark to screen on, but there is
        a time budget, and most losing candidates lose by being slow. Cutting
        them off at a multiple of the incumbent's time turns the worst
        generations from a full timeout into a few extra seconds. Raise
        ``screen_factor`` to switch it off; the cost of it being wrong is one
        discarded candidate, and the log says which.

        *seconds* is that island's incumbent, because the global best's time
        is the wrong bar for an island that is deliberately behind it: every
        candidate on a slower island would be cut off before it finished, and
        the island would never get to show what its approach becomes.
        """
        factor = float(self.cfg.get("screen_factor", 2.0))
        incumbent = seconds or self.best_seconds
        if not incumbent or factor <= 0:
            return self.full_timeout()
        return max(30, min(self.full_timeout(), int(factor * incumbent) + 5))

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

    def archive(self, island: int | None = None) -> list[Candidate]:
        """Correct candidates worth mutating again, on *island* or anywhere.

        The top of the ranking plus the Pareto front: the best score is the
        obvious parent, but a candidate that is slightly worse and much
        smaller is often the one a further change can actually build on.

        The Pareto front is filtered by membership too: it is a property of
        the whole history, and an island that could parent from it would be
        reading the other islands' work without a migration having happened.
        """
        limit = int(self.cfg.get("archive", 12))
        correct = [c for c in self.history
                   if c.correct and c.score is not None and self.resident(c, island)]
        ranked = sorted(correct, key=lambda c: c.score,
                        reverse=self.objective.direction == "maximize")
        keep = ranked[:limit]
        for candidate in self.pareto():
            if candidate not in keep and self.resident(candidate, island):
                keep.append(candidate)
        return keep

    def resident(self, candidate: Candidate, island: int | None) -> bool:
        """Whether *island* may mutate *candidate*. None means any island."""
        return island is None or candidate.generation in self.members[island]

    def sample_parent(self, island: int = 0) -> Candidate:
        """Choose what to mutate on *island*: usually its incumbent, sometimes not.

        Always mutating the best candidate is 1+1 hill climbing, which cannot
        leave a local optimum: once the obvious change is in, every further
        proposal is judged against it and the search stalls. Occasionally
        starting from a worse but different candidate is what makes a second
        basin reachable at all.
        """
        pool = [c for c in self.archive(island) if self.snapshot_exists(c.generation)]
        if len(pool) <= 1 or self.rng.random() < 0.7:
            return self.incumbent(island)
        return self.rng.choice(pool)

    def incumbent(self, island: int) -> Candidate:
        """The candidate *island* promotes against, before any of its own."""
        return self.leaders[island] or self.best

    def migrate(self) -> None:
        """Lend each island's incumbent to the next one round the ring.

        A migrant is made eligible as a parent rather than installed as the
        neighbour's incumbent. Installing it would hand every island the same
        score to beat and undo the separation inside one period, which is the
        convergence islands exist to prevent; offered as a parent it still has
        to produce a child that beats the local incumbent on the benchmark,
        which is the evidence every other promotion needs.
        """
        if self.islands < 2:
            return
        for index, leader in enumerate(self.leaders):
            destination = (index + 1) % self.islands
            if leader is None or leader.generation in self.members[destination]:
                continue
            self.members[destination].add(leader.generation)
            self.migrations += 1
            self.log.info("evolve.migrate", generation=leader.generation,
                          score=leader.score, island=index, to=destination)

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
            # Every island starts from the same working code: the baseline is
            # the one candidate known to be correct, and an island with no
            # parent has nothing to propose against.
            self.leaders = [candidate] * self.islands
            for members in self.members:
                members.add(0)
            self.snapshot(0)
        self.history.append(candidate)
        self.log.info("evolve.baseline", correct=correct, score=candidate.score,
                      noise=round(self.noise, 6), samples=len(candidate.samples))
        self.save()
        return candidate

    # -- candidate generation --------------------------------------------
    def propose(self, generation: int, family: str, parent: Candidate,
                feedback: str = "") -> tuple[dict, list[str]]:
        """Ask for one change, as a patch. Returns the proposal and what it saw.

        *feedback* is what a previous proposal in this same generation came
        back as -- an anchor that matched nothing, or a change already
        measured here. It arrives composed, because what to say about a
        proposal is the caller's knowledge, not this method's.
        """
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
        # No floor: the code is shown whole or not at all. A trimmed copy is
        # the failure this protocol exists to prevent, and it fits by
        # construction anyway -- fill() is given exactly the room it has.
        code = prompt.add("code", priority=10)
        prompt.add("history",
                   (("CHANGES THAT HELPED HERE:\n"
                     + "\n".join(f"- {h}" for h in self.helped[-6:]) + "\n\n")
                    if self.helped else "")
                   + (("CHANGES THAT DID NOT HELP HERE:\n"
                       + "\n".join(f"- {r}" for r in self.hurt[-12:]))
                      if self.hurt else ""), priority=20, floor=400)
        if feedback:
            prompt.add("feedback", feedback, priority=5, floor=200)
        prompt.add("direction", prompts.evolution_direction(family),
                   required=True)
        # The code block takes what the fixed blocks leave, and every target it
        # offers it offers whole: an edit's anchor has to be copied from the
        # file as it actually is, and a fragment is also a candidate reasoning
        # about a hot path whose tail it was never shown.
        sources, shown = self._sources(max(0, code.room() - 14))
        if not shown:
            raise NoTargetFits(
                f"none of {', '.join(self.targets) or '(no targets)'} fits the "
                f"{code.room()} characters this prompt has left for source")
        code.fill("CURRENT CODE:\n" + sources)
        return evolver.ask_json(prompt.render(), schema=prompts.PROPOSAL), shown

    def _sources(self, budget: int) -> tuple[str, list[str]]:
        """Targets that fit *budget* whole, and which ones those are.

        A target too big to show is skipped rather than truncated, and the
        room it did not take is left to the targets after it -- the caller
        confines the candidate to what came back, so a file nobody saw is not
        a file anybody may change.
        """
        sources: list[str] = []
        shown: list[str] = []
        used = 0
        for target in self.targets[:4]:
            text = self.sandbox.read(target)
            if not text:
                continue
            chunk = f"--- {target} ---\n{text}"
            if used + len(chunk) + 2 > budget:
                self.log.warn("evolve.target_too_large", target=target,
                              chars=len(text), room=max(0, budget - used))
                continue
            used += len(chunk) + 2
            sources.append(chunk)
            shown.append(target)
        return "\n\n".join(sources), shown

    def _apply(self, proposal: dict, shown: list[str]) -> Change:
        """Apply a proposal's patch, confined to the targets it was shown.

        The implementer's applier, so an edit whose anchor is missing or
        ambiguous is reported in the same words rather than guessed at. A path
        outside *shown* is refused for the same reason it was not shown:
        rollback here restores target files, so a change anywhere else could
        not be undone when the candidate loses.
        """
        allowed = set(shown)
        payload: dict = {}
        stray: list[str] = []
        for key in ("files", "edits"):
            kept = []
            for item in proposal.get(key) or []:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path", "")).strip().lstrip("/")
                if path in allowed:
                    kept.append(item)
                else:
                    stray.append(path)
            payload[key] = kept
        change = apply_change(self.sandbox, payload, log=self.log)
        for path in dict.fromkeys(stray):
            change.rejected.append(path)
            change.unmatched.append(
                f"{path or '(no path)'}: not one of the files you were shown. "
                f"Change only {', '.join(shown)}.")
        return change

    # -- novelty ----------------------------------------------------------
    def changed_lines(self, before: dict[str, str]) -> frozenset[str]:
        """What a candidate did to its parent, as the lines it moved.

        Not the proposal's own account of itself: the same change arrives as a
        find/replace in one generation and as a whole rewritten file in the
        next, and the mutation line is a label the model chose for it. Blank
        lines and trailing spaces are dropped because they are not a different
        program; indentation is kept, because in an indented language it is.
        """
        changed: set[str] = set()
        for path in sorted(before):
            after = _significant(self.sandbox.read(path) or "")
            # Dropping the first two lines drops the ---/+++ header, which is
            # otherwise indistinguishable from a removed line beginning with a
            # dash. A diff with nothing in it yields no lines at all.
            body = list(difflib.unified_diff(_significant(before[path]), after,
                                             n=0, lineterm=""))[2:]
            changed.update(f"{path}{line}" for line in body if line[:1] in "+-")
        return frozenset(changed)

    def repeat_of(self, parent: int, changes: frozenset[str]) -> Tried | None:
        """The measured change this proposal repeats, if it repeats one.

        A compile-test-benchmark cycle is the most expensive thing the factory
        does, and an evolver told not to repeat itself still re-proposes the
        same edit under a new name. Refusing it before the cycle costs one set
        comparison.

        Scoped to the parent, because the same edit made to two different
        parents is two different programs and only one of them has been
        measured. A patch that changes exactly the lines an earlier one
        changed is always refused, whatever the threshold says: there is
        nothing left to learn from measuring it. The threshold governs the
        near misses, and it compares changed lines rather than patch text
        because a character ratio cannot tell a re-worded repeat from a tuned
        constant -- ``BLOCK = 64`` and ``BLOCK = 128`` differ in one character
        and are two experiments. What it can still be wrong about is a patch
        that moves the same lines somewhere else, which is why a refusal is
        logged and re-asked rather than silently dropped.
        """
        if not changes:
            return None
        threshold = float(self.cfg.get("novelty", 0.8) or 0.0)
        for tried in self.tried:
            if tried.parent != parent:
                continue
            if tried.changes == changes:
                return tried  # the same experiment: its score is already known
            if not 0 < threshold <= 1:
                continue
            if len(tried.changes & changes) / len(tried.changes | changes) >= threshold:
                return tried
        return None

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

        try:
            for generation in range(1, max_generations + 1):
                # Round-robin rather than all islands per generation: the
                # budget is the number of generations, so N islands each get
                # 1/N of it either way, and taking turns spends nothing extra
                # on the ones that are behind.
                island = (generation - 1) % self.islands
                improved_this_round = False
                for _ in range(population):
                    improved_this_round |= self._attempt(len(self.history), island)
                    self.save()
                    if self.objective.reached(self.best.score):
                        self.log.info("evolve.target_reached", score=self.best.score)
                        self.restore(self.best.generation)
                        self.save()
                        return self.best
                if self.migrate_every > 0 and generation % self.migrate_every == 0:
                    self.migrate()
                stagnant = 0 if improved_this_round else stagnant + 1
                if stagnant >= patience:
                    self.log.info("evolve.plateau", generations=generation)
                    break
        except NoTargetFits as exc:
            # Nothing a further generation can change: the targets are a
            # property of the project and the room is a property of the model.
            # Stop and keep the baseline rather than spend the budget on
            # generations that cannot see the code they are meant to improve.
            self.log.warn("evolve.no_visible_target", reason=str(exc)[:200])

        if self.best:
            self.restore(self.best.generation)
        self.save()
        return self.best

    def _draft(self, index: int, family: str, parent: Candidate,
               feedback: str = "") -> tuple[dict, Change, frozenset[str]]:
        """One ask, applied to the parent: the proposal, what landed, what moved.

        An edit that missed makes this a different change from the one
        proposed, so the rest of it goes back too and the whole patch is asked
        for again, told which anchor was wrong. The generation is already paid
        for; a second ask that can see the same code is cheaper than measuring
        half a mutation.
        """
        before = {path: self.sandbox.read(path) or "" for path in self.targets}
        proposal, shown = self.propose(index, family, parent, feedback)
        change = self._apply(proposal, shown)
        if change.unmatched:
            self.log.info("evolve.reanchor", index=index,
                          reason=change.unmatched[0][:160])
            self.restore(parent.generation)
            proposal, shown = self.propose(
                index, family, parent,
                prompts.evolution_reanchor("\n".join(change.unmatched)))
            change = self._apply(proposal, shown)
        return proposal, change, self.changed_lines(before)

    def _attempt(self, index: int, island: int = 0) -> bool:
        """One candidate on *island*: propose, apply, measure, select.

        True when promoted. Everything after the measurement is judged against
        that island's incumbent rather than the global best, which is the
        whole of what an island is: a lineage that started out worse is given
        the generations to become good instead of being discarded on the first
        one for not already beating the leading approach.
        """
        parent = self.sample_parent(island)
        leader = self.incumbent(island)
        family = self.bandit.select()
        feedback = ""
        for _ in range(2):
            self.restore(parent.generation)
            try:
                proposal, change, changes = self._draft(index, family, parent, feedback)
            except NoTargetFits:
                raise
            except Exception as exc:  # noqa: BLE001 - a bad proposal is a bad candidate
                self.log.warn("evolve.propose_failed", index=index, error=str(exc)[:200])
                return False

            mutation = str(proposal.get("mutation", "unnamed"))[:200]
            if change.empty or change.unmatched:
                # Still not a whole patch after the retry: measuring what did
                # land would score a mutation nobody proposed, and promoting it
                # would make it the parent of everything after.
                detail = "; ".join(change.unmatched)[:200] or change.describe()
                self.log.info("evolve.no_change", index=index, detail=detail)
                self.bandit.update(family, 0.0)
                self.restore(parent.generation)
                return False

            seen = self.repeat_of(parent.generation, changes)
            if seen is None:
                break
            self.repeats += 1
            self.log.info("evolve.repeat", index=index, mutation=mutation[:80],
                          repeated=seen.candidate.generation,
                          exact=seen.changes == changes)
            feedback = prompts.evolution_repeat(seen.candidate.mutation, seen.outcome())
        else:
            # Twice the same change out of this family, against this parent, in
            # one generation: it has nothing further to say here, and the
            # bandit is the part of the system that can act on that.
            self.bandit.update(family, 0.0)
            self.restore(parent.generation)
            return False

        correct, score, verification = self.measure(
            timeout=self.screen_timeout(leader.seconds))
        candidate = Candidate(index, parent.generation, mutation, score, correct,
                              change.written, family=family, island=island,
                              samples=[score] if score is not None else [],
                              seconds=self.last_seconds,
                              metrics={"mechanism": proposal.get("mechanism", "")})
        self.history.append(candidate)
        self.members[island].add(index)
        self.tried.append(Tried(parent.generation, changes, candidate))

        if not correct:
            candidate.notes = "failed correctness: " + verification.failure_output(800)
            self.remember(family, mutation, False, "broke tests")
            self.bandit.update(family, 0.0)
            self.log.info("evolve.discard", index=index, reason="incorrect")
            self.restore(parent.generation)
            return False

        self.snapshot(index)  # any correct candidate may become a parent later
        if not self.significant(score, leader.score):
            candidate.notes = f"no significant gain ({score} vs {leader.score})"
            self.remember(family, mutation, False, f"score {score}")
            # Correct but not better is still worth more than broken: it can
            # be mutated again, so the family keeps a fraction of the credit.
            self.bandit.update(family, 0.1 if score is not None else 0.0)
            self.log.info("evolve.discard", index=index, score=score, island=island,
                          best=leader.score, margin=round(self.margin(leader.score), 6))
            self.restore(parent.generation)
            return False

        # It looks better. Measure it again before believing it: the first
        # sample is exactly the one selection is biased towards over-trusting.
        samples = [score] + self.repeat(int(self.cfg.get("baseline_repeats", 3)) - 1)
        candidate.samples = samples
        candidate.score = self.objective.aggregate(samples)
        if not self.significant(candidate.score, leader.score):
            candidate.notes = (f"gain did not reproduce ({score} once, "
                               f"{candidate.score} over {len(samples)} runs)")
            self.remember(family, mutation, False, "gain did not reproduce")
            self.bandit.update(family, 0.1)
            self.log.info("evolve.unconfirmed", index=index, first=score,
                          confirmed=candidate.score, best=leader.score)
            self.restore(parent.generation)
            return False

        self.snapshot(index)  # re-taken: the confirmation runs may have written
        delta = f"{leader.score} -> {candidate.score}"
        self.leaders[island] = candidate
        # The global best is what the run finally restores, so it is held to
        # the same scepticism as an island promotion: an island that is ahead
        # by less than the margin has not been shown to be ahead.
        if self.significant(candidate.score, self.best.score):
            self.best = candidate
            self.best_seconds = candidate.seconds or self.best_seconds
        self.remember(family, mutation, True, f"score {candidate.score}")
        self.bandit.update(family, 1.0)
        self.log.info("evolve.improved", index=index, delta=delta, family=family[:20],
                      island=island, mutation=mutation[:80])
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
        if self.islands > 1:
            lines.append(f"\nislands ({self.migrations} migrations):")
            for index, leader in enumerate(self.leaders):
                born = sum(1 for c in self.history if c.island == index)
                lines.append(f"  island {index}: best={leader.score if leader else None} "
                             f"born={born} may mutate={len(self.members[index])}")
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
