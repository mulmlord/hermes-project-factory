"""Final documentation. Runs after implementation and validation, never before."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .agents import Prompt, context_block
from .log import Logger, NULL
from .sandbox import Sandbox
from .workspace import tree

README_SECTIONS = (
    "purpose", "features", "architecture", "requirements", "installation",
    "configuration", "usage", "examples", "testing", "benchmarks",
    "limitations", "license",
)


def write_all(agents, sandbox: Sandbox, state, board, log: Logger = NULL) -> None:
    readme(agents, sandbox, state, board, log)
    architecture(agents, sandbox, state, log)
    changelog(sandbox.root, state, board)


def _source_digest(sandbox: Sandbox, budget: int = 30000) -> str:
    """Entry points and public surface, so docs describe real behaviour.

    *budget* is the room the caller has left in the documenter's context: the
    README is written from whatever of the project fits here, so the files are
    taken in glob order and each is capped, rather than the first file taking
    everything.
    """
    root = sandbox.root
    interesting = []
    for pattern in ("src/**/*", "*.py", "*.js", "*.ts", "*.go", "*.rs", "*.c", "*.cpp",
                    "Makefile", "CMakeLists.txt", "pyproject.toml", "package.json",
                    "Cargo.toml", "go.mod", "requirements.txt"):
        interesting.extend(sorted(root.glob(pattern)))
    chunks, used = [], 0
    seen = set()
    for path in interesting:
        if not path.is_file() or path in seen or used >= budget:
            continue
        seen.add(path)
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        slice_ = text[: min(max(1, budget // 8), budget - used)]
        used += len(slice_)
        chunks.append(f"--- {path.relative_to(root)} ---\n{slice_}")
    return "\n\n".join(chunks) or "(no sources found)"


def readme(agents, sandbox: Sandbox, state, board, log: Logger = NULL) -> Path:
    commands = state.plan.get("commands") or {}
    completed = [t.title for t in board.by_status("completed")]
    dropped = [f"{t.title} ({t.notes[-1] if t.notes else 'not completed'})"
               for t in board.by_status("abandoned", "blocked")]
    metrics = state.metrics.get("optimization")
    log.info("docs.readme")
    documenter = agents["documenter"]
    prompt = Prompt(documenter, log=log)
    prompt.add("context", context_block(state), required=True)
    prompt.add("tree", f"FILES:\n{tree(sandbox.root)}", priority=40, floor=400)
    prompt.add("spec", f"SPECIFICATION:\n{json.dumps(state.spec, indent=2)}",
               priority=20, floor=1000)
    source = prompt.add("source", priority=10, floor=4000)
    prompt.add("tickets",
               "IMPLEMENTED:\n" + "\n".join(f"- {c}" for c in completed[:40])
               + (("\n\nNOT IMPLEMENTED:\n"
                   + "\n".join(f"- {d}" for d in dropped[:20])) if dropped else ""),
               priority=30, floor=500)
    prompt.add("commands",
               (f"MEASURED PERFORMANCE: {json.dumps(metrics)}\n" if metrics else "")
               + f"COMMANDS: setup=`{commands.get('setup')}` "
                 f"build=`{commands.get('build')}` test=`{commands.get('test')}` "
                 f"benchmark=`{commands.get('benchmark')}`", required=True)
    prompt.add("instructions",
               "Write the project README. Document only what the source above actually "
               "does; never invent a feature or a flag. Every command must be one a user "
               "can really run from a clean checkout. Return JSON with a markdown string "
               "for each key:\n"
               "{" + ", ".join(f'"{s}": ""' for s in README_SECTIONS) + "}\n"
               "purpose: 2-3 sentences. features: a bulleted list of what works today. "
               "architecture: how the pieces fit, with the real file names. examples: at "
               "least one complete runnable example with its expected output. "
               "limitations: honest, including anything listed as not implemented.",
               required=True)
    # The source outranks everything optional: a README written from the spec
    # instead of the code is exactly the invented documentation this forbids.
    source.fill("SOURCE:\n" + _source_digest(sandbox, max(0, source.room() - 8)))
    result = documenter.ask_json(prompt.render())
    title = state.spec.get("name", state.name)
    body = [f"# {title}\n", result.get("purpose", state.idea).strip(), ""]
    for section in README_SECTIONS[1:]:
        content = str(result.get(section, "")).strip()
        if not content:
            continue
        body += [f"## {section.title()}\n", content, ""]
    path = sandbox.root / "README.md"
    path.write_text("\n".join(body).rstrip() + "\n")
    log.info("docs.readme_written", chars=len(path.read_text()))
    return path


def architecture(agents, sandbox: Sandbox, state, log: Logger = NULL) -> Path:
    documenter = agents["documenter"]
    prompt = Prompt(documenter, log=log)
    prompt.add("context", context_block(state), required=True)
    prompt.add("tree", f"FILES:\n{tree(sandbox.root)}", priority=20, floor=400)
    source = prompt.add("source", priority=10, floor=4000)
    prompt.add("instructions",
               "Document the architecture as built. Return JSON:\n"
               '{"overview": "", "components": [{"name": "", "path": "", '
               '"responsibility": "", "interfaces": ""}], "data_flow": "", '
               '"key_decisions": [""], "extension_points": [""]}', required=True)
    source.fill("SOURCE:\n" + _source_digest(sandbox, max(0, source.room() - 8)))
    result = documenter.ask_json(prompt.render())
    lines = [f"# Architecture\n", result.get("overview", ""), "", "## Components\n"]
    for component in result.get("components") or []:
        if not isinstance(component, dict):
            continue
        lines += [f"### {component.get('name', 'component')} "
                  f"(`{component.get('path', '')}`)",
                  component.get("responsibility", ""), ""]
        if component.get("interfaces"):
            lines += [f"**Interfaces:** {component['interfaces']}", ""]
    lines += ["## Data flow\n", result.get("data_flow", ""), ""]
    for title, key in (("Key decisions", "key_decisions"),
                       ("Extension points", "extension_points")):
        items = result.get(key) or []
        if items:
            lines += [f"## {title}\n"] + [f"- {i}" for i in items] + [""]
    path = sandbox.root / "docs" / "architecture" / "overview.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def changelog(root: Path, state, board) -> Path:
    completed = sorted(board.by_status("completed"), key=lambda t: t.updated)
    groups: dict[str, list[str]] = {}
    for ticket in completed:
        groups.setdefault(_section(ticket.kind), []).append(ticket.title)
    lines = ["# Changelog", "",
             "All notable changes to this project.", "",
             f"## [0.1.0] - {time.strftime('%Y-%m-%d')}", ""]
    for section in ("Added", "Changed", "Fixed", "Performance", "Tests", "Documentation"):
        if groups.get(section):
            lines += [f"### {section}", ""] + [f"- {t}" for t in groups[section]] + [""]
    metrics = state.metrics.get("optimization")
    if metrics:
        lines += ["### Performance", "",
                  f"- {metrics['metric']}: {metrics['baseline']} -> {metrics['best']} "
                  f"over {metrics['generations']} generations", ""]
    path = Path(root) / "CHANGELOG.md"
    path.write_text("\n".join(lines))
    return path


def _section(kind: str) -> str:
    return {"test": "Tests", "documentation": "Documentation", "optimization": "Performance",
            "refactor": "Changed", "research": "Documentation"}.get(kind, "Added")
