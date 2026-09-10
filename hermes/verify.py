"""Verification: build, tests, quality gates, review and failure triage."""
from __future__ import annotations

import json
import re
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path

from .log import Logger, NULL
from .sandbox import Result, Sandbox
from .tickets import split_validation

#: Patterns that mark generated code as unsafe regardless of test results.
UNSAFE_PATTERNS = [
    (r"\brm\s+-rf\s+/(?:\s|$|\*)", "recursive delete of the filesystem root"),
    (r"\bcurl\b[^\n|]*\|\s*(?:ba)?sh", "piping a download straight into a shell"),
    (r"\bwget\b[^\n|]*\|\s*(?:ba)?sh", "piping a download straight into a shell"),
    (r"subprocess\.[A-Za-z_]+\([^)]*shell\s*=\s*True[^)]*\+", "shell=True with concatenated input"),
    (r"\beval\s*\(\s*(?:input|request|sys\.argv)", "eval on untrusted input"),
    (r"\bos\.system\s*\([^)]*[+%]", "os.system with interpolated input"),
    (r"~/\.ssh|/\.aws/credentials|id_rsa\b", "reading host credentials"),
    (r"(?i)(api[_-]?key|secret|password)\s*=\s*['\"][A-Za-z0-9/+_-]{16,}", "hardcoded secret"),
]

SOURCE_SUFFIXES = {".py", ".js", ".ts", ".sh", ".rb", ".go", ".rs", ".c", ".cc",
                   ".cpp", ".h", ".hpp", ".java", ".yaml", ".yml", ".toml"}


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False

    def __str__(self) -> str:
        mark = "skip" if self.skipped else ("pass" if self.passed else "FAIL")
        return f"[{mark}] {self.name}" + (f": {self.detail[:200]}" if self.detail else "")


@dataclass
class Verification:
    gates: list[Gate] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed or g.skipped for g in self.gates)

    @property
    def failures(self) -> list[Gate]:
        return [g for g in self.gates if not g.passed and not g.skipped]

    def report(self) -> str:
        return "\n".join(str(g) for g in self.gates)

    def failure_output(self, limit: int = 6000) -> str:
        chunks = [r.summary() for r in self.results if not r.ok]
        return ("\n\n".join(chunks) or self.report())[-limit:]


_MISSING_TOOL = re.compile(r"([\w.+-]+): (?:not found|command not found)")

#: Tools a gate may legitimately need. Anything else named by a model-written
#: command is not fetched from PyPI: that would make typosquatting a supply
#: chain into the sandbox.
DEV_TOOLS = frozenset({
    "pytest", "pytest-cov", "flake8", "ruff", "mypy", "black", "pylint",
    "isort", "coverage", "tox", "hypothesis", "bandit",
})


def _install_missing(sandbox: Sandbox, result, verification: 'Verification',
                     log: Logger) -> bool:
    """Exit 127 means a tool is absent, not that the generated code is wrong.

    The architect is meant to install its tooling in the setup command and
    routinely forgets, which otherwise fails every gate for the rest of the run.
    """
    found = _MISSING_TOOL.search(result.stderr + result.stdout)
    if found:
        tool = found.group(1)
    elif result.exit_code == 127:
        # A pipeline hides 127 behind the exit code of its last stage, so the
        # message above is the reliable signal and this is only the fallback.
        tool = (shlex.split(result.command or "")[:1] or [""])[0]
    else:
        return False
    if tool in ("", "sh", "bash", "python", "python3"):
        return False
    if sandbox.runtime != "docker":
        return False  # installing onto the host is not ours to do
    if tool in DEV_TOOLS:
        target = tool
    elif any((sandbox.root / f).is_file() for f in ("pyproject.toml", "setup.py")):
        target = "-e ."  # a missing console script means the project is uninstalled
    else:
        return False
    log.info("verify.install_missing", tool=tool, install=target)
    attempt = sandbox.install(f"pip install --quiet {target}")
    if not attempt.ok:
        # Why the install failed is the actual diagnosis (a src/ layout with no
        # package_dir, say), so it has to reach the agent that must fix it.
        verification.results.append(attempt)
        return False
    sandbox.installed.append(target)
    return True


def run_commands(sandbox: Sandbox, commands: list[str], name: str,
                 verification: Verification, *, required: bool = True,
                 timeout: int | None = None, install: bool = False,
                 autofix: bool = True, log: Logger = NULL) -> None:
    """Run a command list and fold the outcome into *verification* as one gate."""
    commands = [c for c in commands if c and c.strip() and c.strip() != "true"]
    if not commands:
        verification.gates.append(Gate(name, True, "no command configured", skipped=True))
        return
    for command in commands:
        runner = sandbox.install if install else sandbox.run
        result = runner(command, timeout=timeout)
        if not result.ok and autofix and _install_missing(sandbox, result, verification, log):
            result = runner(command, timeout=timeout)
        verification.results.append(result)
        if not result.ok:
            verification.gates.append(
                Gate(name, not required, result.summary(1500), skipped=not required))
            log.warn("verify.failed", gate=name, command=command[:80],
                     exit=result.exit_code)
            return
    verification.gates.append(Gate(name, True, f"{len(commands)} command(s) ok"))


def scan_unsafe(root: Path, paths: list[str] | None = None) -> list[str]:
    """Static security scan over generated sources. Returns human-readable hits."""
    root = Path(root)
    targets = ([root / p for p in paths] if paths else
               [p for p in root.rglob("*")
                if p.suffix in SOURCE_SUFFIXES
                and not any(part in (".git", ".hermes", "node_modules", ".venv",
                                     "artifacts", "__pycache__")
                            for part in p.parts)])
    hits = []
    for path in targets:
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for pattern, description in UNSAFE_PATTERNS:
            for match in re.finditer(pattern, text):
                line = text[: match.start()].count("\n") + 1
                rel = path.relative_to(root) if root in path.parents else path
                hits.append(f"{rel}:{line}: {description}")
    return hits


def record_installs(sandbox: Sandbox, plan: dict) -> None:
    """Fold anything Hermes had to install into the project's own setup command.

    Otherwise the clean-room check would fail on a dependency that only exists
    because Hermes quietly installed it, and the published repo would not build.
    """
    commands = plan.setdefault("commands", {})
    setup = (commands.get("setup") or "").strip()
    pending = [t for t in sandbox.installed if t not in setup]
    if not pending:
        return
    added = "pip install " + " ".join(pending)
    commands["setup"] = added if setup in ("", "true") else f"{setup} && {added}"


def verify_project(sandbox: Sandbox, plan: dict, *, include_benchmark: bool = False,
                   log: Logger = NULL) -> Verification:
    """The full project gate: setup, build, lint, tests, security."""
    commands = plan.get("commands") or {}
    verification = Verification()
    run_commands(sandbox, [commands.get("setup", "")], "setup", verification,
                 required=False, install=True, log=log)
    run_commands(sandbox, [commands.get("build", "")], "build", verification, log=log)
    run_commands(sandbox, [commands.get("lint", "")], "lint", verification,
                 required=False, log=log)
    run_commands(sandbox, [commands.get("test", "")], "tests", verification, log=log)
    if include_benchmark and commands.get("benchmark"):
        run_commands(sandbox, [commands["benchmark"]], "benchmark", verification,
                     required=False, log=log)
    hits = scan_unsafe(sandbox.root)
    verification.gates.append(
        Gate("security", not hits, "; ".join(hits[:5]) if hits else "no unsafe patterns"))
    record_installs(sandbox, plan)
    return verification


def verify_ticket(sandbox: Sandbox, ticket, plan: dict, changed: list[str],
                  log: Logger = NULL, tests_required: bool = True) -> Verification:
    """Gate a single ticket: its own validation commands plus the project tests.

    Early in a project the test suite does not exist yet, so failing it would
    block the very ticket that creates it. Until the suite has passed once,
    it is advisory; from then on it is a hard regression gate.
    """
    commands = plan.get("commands") or {}
    verification = Verification()
    run_commands(sandbox, [commands.get("build", "")], "build", verification, log=log)
    # Filtered here too, not just at planning: tickets created before the check
    # exists still carry prose that would fail as a command.
    runnable, _ = split_validation(ticket.validation)
    run_commands(sandbox, runnable, f"ticket:{ticket.id}", verification, log=log)
    run_commands(sandbox, [commands.get("test", "")], "tests", verification,
                 required=tests_required, log=log)
    hits = scan_unsafe(sandbox.root, changed)
    verification.gates.append(
        Gate("security", not hits, "; ".join(hits[:5]) if hits else "clean"))
    record_installs(sandbox, plan)
    return verification


def review(agents, ticket, diff: str, verification: Verification,
           log: Logger = NULL) -> dict:
    """Reviewer verdict on a ticket's change. Blocking issues send it back."""
    verdict = agents["reviewer"].ask_json(
        f"TICKET {ticket.id}: {ticket.title}\n"
        f"OBJECTIVE: {ticket.objective}\n"
        f"ACCEPTANCE CRITERIA:\n" + "\n".join(f"- {a}" for a in ticket.acceptance) +
        f"\n\nVERIFICATION RESULTS:\n{verification.report()}\n\n"
        f"CHANGE:\n{diff[:20000]}\n\n"
        "Review this change. Return JSON:\n"
        "{\n"
        '  "approved": true|false,\n'
        '  "meets_acceptance": true|false,\n'
        '  "blocking": ["issue that must be fixed before this can be accepted"],\n'
        '  "advisory": ["worth improving but not blocking"],\n'
        '  "security": ["security concern, or empty"],\n'
        '  "reason": "one sentence verdict"\n'
        "}\n"
        "Block only on real defects: incorrect behaviour, unmet acceptance "
        "criteria, stubbed-out work, security problems, or missing error "
        "handling at a trust boundary. Do not block on style or taste."
    )
    blocking = [b for b in (verdict.get("blocking") or []) if str(b).strip()]
    security = [s for s in (verdict.get("security") or []) if str(s).strip()]
    verdict["approved"] = bool(verdict.get("approved")) and not blocking and not security
    verdict["blocking"] = blocking + security
    log.info("review.verdict", ticket=ticket.id, approved=verdict["approved"],
             blocking=len(verdict["blocking"]))
    return verdict


def triage(agents, ticket, verification: Verification, context: str,
           attempts: int, log: Logger = NULL) -> dict:
    """Classify a failure and prescribe the smallest repair."""
    verdict = agents["triager"].ask_json(
        f"{context}\n\nTICKET {ticket.id}: {ticket.title}\n"
        f"ATTEMPT {attempts} of this ticket.\n\n"
        f"GATES:\n{verification.report()}\n\n"
        f"OUTPUT:\n{verification.failure_output()}\n\n"
        "Diagnose the root cause. Return JSON:\n"
        "{\n"
        '  "diagnosis": "what is actually wrong, not what the error text says",\n'
        '  "classification": "trivial|implementation|architecture|research_gap|'
        'environment|flaky",\n'
        '  "action": "retry|repair|decompose|reconsider|research|rollback",\n'
        '  "instructions": "exactly what the next agent must change",\n'
        '  "files": ["file that needs changing"],\n'
        '  "research_topic": "question to research, only when action=research"\n'
        "}\n"
        "trivial = a typo, a missing import, a wrong path. implementation = the "
        "code is wrong. architecture = the design cannot work. research_gap = we "
        "do not know enough to fix it. environment = a missing tool or dependency."
    )
    log.info("triage", ticket=ticket.id, classification=verdict.get("classification"),
             action=verdict.get("action"))
    return verdict


def record_evaluation(root: Path, label: str, verification: Verification,
                      extra: dict | None = None) -> Path:
    path = (Path(root) / ".hermes" / "evaluations" /
            f"{time.strftime('%Y%m%d-%H%M%S')}-{label}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "label": label,
        "at": time.time(),
        "passed": verification.passed,
        "gates": [{"name": g.name, "passed": g.passed, "skipped": g.skipped,
                   "detail": g.detail[:2000]} for g in verification.gates],
        **(extra or {}),
    }, indent=2, default=str))
    return path
