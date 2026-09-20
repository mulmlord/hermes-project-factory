"""Section 35: the factory improving its own components.

Everything else in Hermes changes a generated project. This changes Hermes,
which is a different kind of act and needs a different kind of care: the
program being modified is the program doing the modifying, and the gate that
judges a change is written in the same tree the change can reach.

Three properties make that safe enough to attempt, and each is enforced here
rather than asked for in a prompt.

*The candidate never touches the running installation.* A proposal is
materialised as a git worktree -- its own directory on its own branch -- so
the tree Hermes is executing from is untouched while the candidate is built
and judged. Switching the live checkout's branch instead would rewrite the
source of the running process, which is precisely the failure this avoids.

*The candidate cannot edit its own judge.* :data:`PROTECTED` names the test
suite, the measurement harness and this module. A change that touches any of
them is refused before it is evaluated, and the trusted copies are restored
into the worktree before the suite runs, so the tests that pass judgement are
always ours. Without that rule the cheapest way to pass is to delete the
tests, and a loop that can do that will eventually do it.

*An unmeasured change is not an improvement.* Section 35 permits promotion
only on demonstrated benefit, so the benefit gate refuses by default and must
be shown a before/after evaluation to open. A host with no local model cannot
produce one, and on such a host nothing is promotable -- which is the correct
outcome, not a limitation to work around.

Promotion is operator-invoked (``hermes improve``) and never runs inside the
autonomous project loop: a factory building someone's project has no business
rewriting itself halfway through.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .agents import Prompt
from .log import Logger, NULL
from .verify import Gate

#: Paths a self-modification may never touch, because they are what decides
#: whether a self-modification is any good. Matched as path prefixes against
#: the repository root.
PROTECTED = (
    "tests",              # the suite that says pass or fail
    "hermes/improve.py",  # this gate
    "hermes/harness.py",  # the measurement the benefit gate reads
    "conftest.py",
    "pyproject.toml",     # testpaths, dependencies, entry points
    "install.sh",
    "uninstall.sh",
    ".github",
    ".git",
)

#: The components section 35 names, mapped to the files that implement them.
#: Benchmark infrastructure is absent on purpose: it lives in harness.py,
#: which is protected, so the factory cannot rewrite the instrument it is
#: judged by. That is a real restriction and it is the right one.
TARGETS: dict[str, tuple[str, ...]] = {
    "planner": ("hermes/planning.py",),
    "decomposition": ("hermes/planning.py",),
    "routing": ("hermes/models.py",),
    "prompts": ("hermes/agents.py",),
    "evaluation": ("hermes/verify.py",),
    "candidates": ("hermes/evolve.py",),
    "scheduling": ("hermes/tickets.py",),
    "caching": ("hermes/models.py",),
}


def protects(path: str) -> bool:
    """Whether *path* is part of what judges a change."""
    posix = Path(path).as_posix().lstrip("./")
    return any(posix == p or posix.startswith(p.rstrip("/") + "/")
               for p in PROTECTED)


def admissible(paths) -> str:
    """The reason a change may not be evaluated, or "" if it may.

    Refusing here, before anything runs, is deliberate: a proposal that
    rewrites the tests is not a failing candidate to be scored, it is a
    candidate that must never be scored at all.
    """
    if not paths:
        return "the proposal changes no files"
    guarded = sorted({p for p in paths if protects(p)})
    if guarded:
        return ("refuses to evaluate a change to what judges it: "
                + ", ".join(guarded))
    outside = sorted({p for p in paths if not Path(p).as_posix().startswith("hermes/")})
    if outside:
        return "changes files outside the hermes package: " + ", ".join(outside)
    return ""


def measured_gain(before: Path, after: Path) -> tuple[bool, str]:
    """Whether two evaluation runs show the change actually helped.

    Deliberately strict. Completing fewer tasks is a regression whatever else
    improved, a pair of stub runs proves nothing about a factory, and a pair
    that moved nowhere is not evidence of benefit -- it is evidence of no
    detectable effect, which section 35 does not accept as grounds to promote.
    """
    def load(path):
        data = json.loads(Path(path).read_text())
        return {t["name"]: t for t in data.get("trials", [])}

    try:
        old, new = load(before), load(after)
    except (OSError, ValueError) as exc:
        return False, f"cannot read the evaluations: {exc}"

    shared = sorted(set(old) & set(new))
    if not shared:
        return False, "the two runs share no task"
    if not any(t.get("model_calls") for t in list(old.values()) + list(new.values())):
        return False, "neither run called a model: this compares two stubs"

    done_before = sum(1 for n in shared if old[n].get("completed"))
    done_after = sum(1 for n in shared if new[n].get("completed"))
    if done_after < done_before:
        return False, f"completed {done_after}/{len(shared)}, was {done_before}"

    def weighted(runs):
        weight = sum(runs[n].get("tickets_completed", 0) for n in shared)
        if not weight:
            return 0.0
        return sum(runs[n].get("first_pass", 0.0) * runs[n].get("tickets_completed", 0)
                   for n in shared) / weight

    pass_before, pass_after = weighted(old), weighted(new)
    if done_after > done_before:
        return True, (f"completed {done_after}/{len(shared)}, was {done_before}; "
                      f"first-pass {pass_before:.0%} -> {pass_after:.0%}")
    if pass_after > pass_before + 0.01:
        return True, (f"first-pass {pass_before:.0%} -> {pass_after:.0%} "
                      f"at {done_after}/{len(shared)} completed")
    return False, (f"no gain: completed {done_after}/{len(shared)}, "
                   f"first-pass {pass_before:.0%} -> {pass_after:.0%}")


@dataclass
class Attempt:
    """One proposed self-modification and everything that judged it."""

    target: str
    branch: str = ""
    rationale: str = ""
    files: list[str] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)
    promoted: bool = False
    refusal: str = ""
    at: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(g.passed for g in self.gates)

    def report(self) -> str:
        head = f"{self.target}: {self.rationale or '(no rationale given)'}"
        body = [str(g) for g in self.gates] or ["(never evaluated)"]
        if self.refusal:
            body.append(f"[FAIL] admissible: {self.refusal}")
        tail = "promoted" if self.promoted else "not promoted"
        return "\n".join([head, *body, tail])


class Improvement:
    """Propose, evaluate and promote a change to Hermes itself."""

    def __init__(self, root: Path, agents: dict | None = None, log: Logger = NULL,
                 python: str | None = None, timeout: int = 1800):
        self.root = Path(root).resolve()
        self.agents = agents or {}
        self.log = log
        self.python = python or sys.executable
        self.timeout = timeout
        self.scratch = self.root / ".hermes-improve"

    # -- git ---------------------------------------------------------------
    def _git(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=str(cwd or self.root),
                              capture_output=True, text=True)

    def dirty(self) -> bool:
        """Uncommitted work in the live tree, which promotion must not bury."""
        return bool(self._git("status", "--porcelain").stdout.strip())

    def worktree(self, branch: str) -> Path:
        """Materialise *branch* in its own directory, off the current HEAD."""
        self.scratch.mkdir(parents=True, exist_ok=True)
        path = self.scratch / branch.replace("/", "-")
        self.discard(branch)
        proc = self._git("worktree", "add", "-q", "-b", branch, str(path), "HEAD")
        if proc.returncode != 0:
            raise RuntimeError(f"git worktree: {proc.stderr.strip()[:300]}")
        return path

    def discard(self, branch: str) -> None:
        """Remove a candidate worktree and its branch. Promotion is the only
        thing that survives an attempt."""
        path = self.scratch / branch.replace("/", "-")
        self._git("worktree", "remove", "--force", str(path))
        shutil.rmtree(path, ignore_errors=True)
        self._git("worktree", "prune")
        self._git("branch", "-D", branch)

    # -- candidate ---------------------------------------------------------
    def apply(self, worktree: Path, files: list[dict]) -> list[str]:
        """Write the proposed files into the worktree. Escapes are refused.

        Refused, not repaired: stripping the leading slash off an absolute
        path would turn ``/etc/x`` into a file the candidate never asked for
        and quietly accept a proposal that aimed outside the tree. On this
        path a rejection is always safer than a guess, and admissible() has
        already had its say, so anything arriving here that still looks wrong
        is dropped rather than interpreted.
        """
        written = []
        for entry in files or []:
            if not isinstance(entry, dict) or not entry.get("path"):
                continue
            raw = str(entry["path"]).strip()
            relative = Path(raw)
            if relative.is_absolute() or ".." in relative.parts or protects(raw):
                continue
            target = (worktree / relative).resolve()
            if not str(target).startswith(str(worktree.resolve())):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(entry.get("content", "")))
            written.append(relative.as_posix())
        return written

    def restore_trusted(self, worktree: Path) -> None:
        """Put our copy of everything that judges a change back.

        Belt and braces with :func:`admissible`: that refuses a proposal which
        names a protected path, and this undoes one that reached the worktree
        by any other route. The suite about to run is ours either way.
        """
        for relative in PROTECTED:
            if relative in (".git", ".github"):
                continue
            source = self.root / relative
            target = worktree / relative
            if source.is_dir():
                shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(source, target,
                                ignore=shutil.ignore_patterns("__pycache__"))
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    # -- gates -------------------------------------------------------------
    def suite(self, worktree: Path) -> Gate:
        """Run the Hermes test suite against the candidate's own source."""
        resolved = self._resolves_to(worktree)
        if resolved is not True:
            return Gate("imports-candidate", False,
                        f"the suite would test {resolved} instead of the "
                        "candidate, so its result would be meaningless")
        proc = self._run([self.python, "-m", "pytest", "-q", "-p",
                          "no:cacheprovider"], worktree)
        if proc is None:
            return Gate("suite", False, f"timed out after {self.timeout}s")
        detail = (proc.stdout or proc.stderr).strip().splitlines()
        return Gate("suite", proc.returncode == 0,
                    detail[-1][:200] if detail else "no output")

    def _resolves_to(self, worktree: Path):
        """True when `import hermes` in *worktree* finds the candidate.

        An editable install puts a finder on sys.meta_path that can shadow
        PYTHONPATH, so which copy the suite imports is a question to answer
        rather than assume: testing the live tree and calling it a candidate
        result would make the whole gate a decoration.
        """
        proc = self._run([self.python, "-c",
                          "import hermes; print(hermes.__file__)"], worktree)
        if proc is None or proc.returncode != 0:
            return "nothing importable"
        found = Path(proc.stdout.strip())
        return True if str(found).startswith(str(worktree.resolve())) else str(found)

    def benefit(self, before: Path | None, after: Path | None) -> Gate:
        """Measured improvement, refused unless it was actually measured."""
        if not before or not after:
            return Gate(
                "benefit", False,
                "no before/after evaluation supplied. Section 35 permits "
                "promotion only on demonstrated benefit, and an unmeasured "
                "change is not demonstrated. Run `hermes eval` on each side "
                "and pass both results.json files.")
        improved, detail = measured_gain(Path(before), Path(after))
        return Gate("benefit", improved, detail)

    def _run(self, argv: list[str], cwd: Path):
        env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/tmp",
               "PYTHONPATH": str(cwd), "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            return subprocess.run(argv, cwd=str(cwd), capture_output=True,
                                  text=True, timeout=self.timeout, env=env)
        except subprocess.TimeoutExpired:
            return None

    # -- the loop ----------------------------------------------------------
    def attempt(self, target: str, proposal: dict | None = None,
                before: Path | None = None, after: Path | None = None) -> Attempt:
        """Build a candidate for *target* and judge it. Promotes nothing."""
        if target not in TARGETS:
            return Attempt(target=target, refusal=f"unknown target {target!r}")
        proposal = proposal if proposal is not None else self.propose(target)
        record = Attempt(target=target,
                         rationale=str(proposal.get("rationale", ""))[:500])

        paths = [str(f.get("path", "")) for f in (proposal.get("files") or [])
                 if isinstance(f, dict)]
        record.refusal = admissible(paths)
        if record.refusal:
            self.log.warn("improve.refused", target=target, reason=record.refusal)
            return record

        branch = f"hermes/improve-{target}-{int(record.at)}"
        worktree = self.worktree(branch)
        record.branch = branch
        try:
            record.files = self.apply(worktree, proposal.get("files") or [])
            if not record.files:
                # Every path was dropped by apply() -- an escape, an absolute
                # path, a protected file -- so the candidate is byte for byte
                # the current tree. Judging it would pass the suite on the
                # strength of code nobody changed, and a before/after pair
                # handed in by the operator would then promote an empty commit
                # as though it were the improvement those numbers measured.
                record.refusal = ("no proposed file survived path checking, "
                                  "so the candidate is the current tree")
                self.log.warn("improve.refused", target=target,
                              reason=record.refusal)
                self.discard(branch)
                record.branch = ""
                return record
            self.restore_trusted(worktree)
            record.gates = [self.suite(worktree), self.benefit(before, after)]
            if record.passed:
                self._git("add", "-A", cwd=worktree)
                self._git("commit", "-q", "-m",
                          f"perf: improve {target}\n\n{record.rationale}",
                          cwd=worktree)
        except Exception as exc:  # noqa: BLE001 - a broken candidate is a result
            record.gates = [Gate("candidate", False, f"{type(exc).__name__}: {exc}")]
        self.log.info("improve.evaluated", target=target, passed=record.passed,
                      files=len(record.files))
        return record

    def propose(self, target: str) -> dict:
        """Ask the implementer for a change to one of Hermes' own components."""
        agent = self.agents.get("implementer")
        if agent is None:
            return {}
        prompt = Prompt(agent, log=self.log)
        prompt.add("task",
                   f"You are improving Hermes itself: the {target} component.\n\n"
                   "Propose one concrete, self-contained improvement. It must keep "
                   "every existing test passing, change only the files shown, and be "
                   "justified by something measurable -- fewer model calls, a higher "
                   "first-attempt gate pass rate, less wall clock. Do not change "
                   "tests, the evaluation harness, or packaging: those judge your "
                   "change and are rejected on sight.\n\n"
                   "Return JSON: {\"rationale\": str, \"files\": [{\"path\": str, "
                   "\"content\": str}]} with each file complete.", required=True)
        sources = prompt.add("sources", priority=10, floor=2000)
        # A component is several files and the answer must return them whole,
        # so each gets an equal share of what is left rather than the first
        # one taking the budget and the rest arriving as stubs.
        share = sources.room() // max(1, len(TARGETS[target]))
        sources.fill("\n\n".join(
            f"--- {relative} ---\n{(self.root / relative).read_text()[:share]}"
            for relative in TARGETS[target] if (self.root / relative).is_file()))
        answer = agent.ask_json(prompt.render())
        return answer if isinstance(answer, dict) else {}

    def promote(self, record: Attempt) -> bool:
        """Merge a passing candidate into the live branch, and prove it there.

        A candidate is judged alone; promotion puts it beside everything else,
        and the suite is run again afterwards because a change can pass in
        isolation and break in company. That re-run is what makes the gate
        unbypassable in practice: nothing lands that the live tree rejects.
        """
        if not record.passed or not record.branch:
            self.log.warn("improve.not_promotable", target=record.target)
            return False
        if self.dirty():
            self.log.warn("improve.dirty_tree", target=record.target)
            return False
        head = self._git("rev-parse", "HEAD").stdout.strip()
        merge = self._git("merge", "--ff-only", record.branch)
        if merge.returncode != 0:
            self.log.warn("improve.merge_failed",
                          error=merge.stderr.strip()[:200])
            return False
        gate = self.suite(self.root)
        if not gate.passed:
            self._git("reset", "--hard", head)
            self.log.warn("improve.reverted", target=record.target,
                          detail=gate.detail[:200])
            return False
        record.promoted = True
        self.log.info("improve.promoted", target=record.target, files=record.files)
        return True
