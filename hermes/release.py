"""Clean-room release validation.

The point is to prove the repository is reproducible: export a pristine copy
of the tracked tree into a fresh sandbox and run setup, build and tests there,
with no state left over from the build that produced it.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .log import Logger, NULL
from .sandbox import Sandbox, Result
from .verify import Gate, Verification, run_commands, scan_unsafe

#: Never shipped in a release: factory internals and evolution scratch.
EXCLUDE = (".hermes", "artifacts", ".git", "__pycache__", ".pytest_cache", ".venv")


def export(root: Path, dest: Path) -> list[str]:
    """Copy the releasable tree into *dest*. Returns the top-level entries."""
    dest.mkdir(parents=True, exist_ok=True)
    tracked = _tracked(root)
    if tracked:
        for relative in tracked:
            source = root / relative
            if not source.is_file() or any(part in EXCLUDE for part in Path(relative).parts):
                continue
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    else:
        shutil.copytree(root, dest, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(*EXCLUDE))
    return sorted(p.name for p in dest.iterdir())


def _tracked(root: Path) -> list[str]:
    proc = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True)
    return proc.stdout.split() if proc.returncode == 0 else []


def validate_clean_room(sandbox: Sandbox, state, log: Logger = NULL) -> tuple[Verification, str]:
    """Run the project's own commands against a fresh export of itself."""
    commands = state.plan.get("commands") or {}
    verification = Verification()
    staging = Path(tempfile.mkdtemp(prefix="hermes-release-"))
    log.info("release.clean_room", staging=str(staging))
    try:
        entries = export(sandbox.root, staging)
        clean = Sandbox(staging, dict(sandbox.cfg), log)
        with clean:
            run_commands(clean, [commands.get("setup", "")], "clean-room-setup",
                         verification, required=False, install=True, log=log)
            run_commands(clean, [commands.get("build", "")], "build",
                         verification, log=log)
            run_commands(clean, [commands.get("test", "")], "tests",
                         verification, log=log)
        missing = [name for name in ("README.md", ".gitignore")
                   if not (staging / name).is_file()]
        verification.gates.append(
            Gate("release-files", not missing,
                 f"missing {missing}" if missing else f"{len(entries)} top-level entries"))
        leaked = scan_unsafe(staging)
        verification.gates.append(
            Gate("release-security", not leaked, "; ".join(leaked[:5]) or "clean"))
        verification.gates.append(
            Gate("clean-room", verification.passed,
                 "pristine export built and tested from scratch"))
        report = _report(entries, verification)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log.info("release.validated", passed=verification.passed)
    return verification, report


def _report(entries: list[str], verification: Verification) -> str:
    return (
        "# Release validation (clean room)\n\n"
        "A pristine export of the tracked tree was built and tested in a fresh "
        "sandbox with no leftover state.\n\n"
        "## Exported top level\n"
        + "\n".join(f"- `{e}`" for e in entries)
        + "\n\n## Gates\n```\n" + verification.report() + "\n```\n"
        + ("\n## Failures\n```\n" + verification.failure_output(4000) + "\n```\n"
           if not verification.passed else "\nAll gates passed.\n")
    )
