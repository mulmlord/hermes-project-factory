"""Project state machine, persistence and checkpointing."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

PHASES = [
    "INTAKE",
    "RESEARCH",
    "SPECIFICATION",
    "PLANNING",
    "IMPLEMENTATION",
    "VALIDATION",
    "OPTIMIZATION",
    "DOCUMENTATION",
    "RELEASE",
    "COMPLETED",
]
TERMINAL = {"COMPLETED", "FAILED"}

#: Allowed forward transitions. Any phase may fail, and validation may bounce
#: back to implementation when the project does not hold together yet.
TRANSITIONS: dict[str, set[str]] = {
    "INTAKE": {"RESEARCH", "SPECIFICATION"},
    "RESEARCH": {"SPECIFICATION"},
    "SPECIFICATION": {"PLANNING"},
    "PLANNING": {"IMPLEMENTATION"},
    "IMPLEMENTATION": {"VALIDATION", "RESEARCH", "PLANNING"},
    "VALIDATION": {"IMPLEMENTATION", "OPTIMIZATION", "DOCUMENTATION"},
    "OPTIMIZATION": {"VALIDATION", "DOCUMENTATION"},
    "DOCUMENTATION": {"RELEASE"},
    "RELEASE": {"COMPLETED", "IMPLEMENTATION"},
    "COMPLETED": set(),
    "FAILED": {"IMPLEMENTATION", "PLANNING"},
}


class StateError(RuntimeError):
    pass


@dataclass
class State:
    idea: str = ""
    name: str = "project"
    phase: str = "INTAKE"
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    iteration: int = 0
    paused: bool = False
    spec: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    failures: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # -- persistence -----------------------------------------------------
    @staticmethod
    def _file(root: Path) -> Path:
        return Path(root) / ".hermes" / "state" / "state.json"

    @classmethod
    def load(cls, root: Path) -> "State":
        path = cls._file(root)
        if not path.is_file():
            raise StateError(f"no Hermes project at {root} (missing {path})")
        raw = json.loads(path.read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, root: Path) -> None:
        self.updated = time.time()
        path = self._file(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, default=str))
        tmp.replace(path)  # atomic: a crash mid-write cannot corrupt state

    # -- state machine ---------------------------------------------------
    def transition(self, phase: str, reason: str = "") -> None:
        if phase not in PHASES and phase != "FAILED":
            raise StateError(f"unknown phase {phase!r}")
        if phase != self.phase and phase not in TRANSITIONS[self.phase] and phase != "FAILED":
            raise StateError(f"illegal transition {self.phase} -> {phase}")
        self.history.append(
            {"from": self.phase, "to": phase, "at": time.time(), "reason": reason}
        )
        self.phase = phase

    def record_failure(self, key: str) -> int:
        self.failures[key] = self.failures.get(key, 0) + 1
        return self.failures[key]

    def clear_failure(self, key: str) -> None:
        self.failures.pop(key, None)

    @property
    def done(self) -> bool:
        return self.phase in TERMINAL


def checkpoint(root: Path, label: str, keep: int = 20) -> Path:
    """Snapshot machine state + tickets so a crash cannot lose progress."""
    root = Path(root)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)[:48]
    dest = root / ".hermes" / "checkpoints" / f"{stamp}-{safe}"
    dest.mkdir(parents=True, exist_ok=True)
    src = root / ".hermes" / "state"
    if src.is_dir():
        shutil.copytree(src, dest / "state", dirs_exist_ok=True)
    _prune(dest.parent, keep)
    return dest


def _prune(directory: Path, keep: int) -> None:
    entries = sorted((p for p in directory.iterdir() if p.is_dir()), key=lambda p: p.name)
    for stale in entries[:-keep] if len(entries) > keep else []:
        shutil.rmtree(stale, ignore_errors=True)


def restore(root: Path, checkpoint_dir: Path) -> None:
    src = Path(checkpoint_dir) / "state"
    if not src.is_dir():
        raise StateError(f"checkpoint has no state directory: {checkpoint_dir}")
    shutil.copytree(src, Path(root) / ".hermes" / "state", dirs_exist_ok=True)
