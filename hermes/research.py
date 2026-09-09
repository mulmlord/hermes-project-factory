"""Feynman-style research workflow.

The loop is: explain the problem simply, name the unknowns, attack them,
challenge the resulting assumptions, and convert what survives into
decisions, hypotheses and tickets. Output must change what gets built --
a summary that changes nothing is treated as a failed round.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .log import Logger, NULL
from .workspace import slugify


@dataclass
class Unknown:
    id: str
    question: str
    why_it_matters: str = ""
    resolution: str = ""
    confidence: str = "unknown"      # unknown | inferred | verified
    needs_experiment: bool = False


@dataclass
class Finding:
    question: str
    answer: str
    confidence: str
    implications: list[str] = field(default_factory=list)
    source: str = "model-knowledge"


class ResearchLog:
    """Append-only record under research/ and .hermes/state/research.json."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / ".hermes" / "state" / "research.json"
        self.rounds: list[dict[str, Any]] = []
        if self.path.is_file():
            self.rounds = json.loads(self.path.read_text()).get("rounds", [])

    def add(self, topic: str, payload: dict) -> None:
        self.rounds.append({"topic": topic, "at": time.time(), **payload})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"rounds": self.rounds}, indent=2, default=str))

    def known(self) -> str:
        """Everything established so far, for feeding back into later rounds."""
        lines = []
        for round_ in self.rounds:
            for finding in round_.get("findings", []):
                lines.append(f"- ({finding.get('confidence', '?')}) "
                             f"{finding.get('question')}: {finding.get('answer')}")
        return "\n".join(lines[-40:]) or "(nothing established yet)"


def investigate(agents, root: Path, topic: str, context: str, *,
                max_unknowns: int = 6, log: Logger = NULL) -> dict:
    """Run one Feynman round on *topic*. Returns findings and actions."""
    researcher = agents["researcher"]
    record = ResearchLog(root)
    log.info("research.round", topic=topic[:80])

    plan = researcher.ask_json(
        f"{context}\n\nRESEARCH TOPIC: {topic}\n\n"
        f"ALREADY ESTABLISHED:\n{record.known()}\n\n"
        "Step 1 of the Feynman method. Return JSON:\n"
        "{\n"
        '  "simple_explanation": "explain the problem as if to a smart person '
        'outside the field, in 3-5 sentences",\n'
        '  "what_we_know": ["established facts that need no research"],\n'
        f'  "unknowns": [{{"question": "specific answerable question", '
        '"why_it_matters": "what decision it changes", '
        '"needs_experiment": true|false}}],\n'
        '  "assumptions_to_challenge": ["assumption that could be wrong"]\n'
        "}\n"
        f"List at most {max_unknowns} unknowns, ordered by how much each would "
        "change the implementation. Skip anything already established."
    )
    unknowns = (plan.get("unknowns") or [])[:max_unknowns]

    findings = []
    for index, unknown in enumerate(unknowns, 1):
        question = unknown.get("question") if isinstance(unknown, dict) else str(unknown)
        if not question:
            continue
        log.info("research.unknown", n=index, question=question[:80])
        answer = researcher.ask_json(
            f"{context}\n\nQUESTION: {question}\n"
            f"WHY IT MATTERS: {unknown.get('why_it_matters', '') if isinstance(unknown, dict) else ''}\n\n"
            "Answer from what you actually know. Return JSON:\n"
            "{\n"
            '  "answer": "the substantive answer, with concrete names, algorithms, '
            'complexities, formats or numbers where they exist",\n'
            '  "confidence": "verified|inferred|uncertain",\n'
            '  "implications": ["what this means for the implementation"],\n'
            '  "alternatives": [{"option": "name", "pros": "", "cons": ""}],\n'
            '  "recommendation": "which alternative to take and why",\n'
            '  "experiment": "a measurable experiment that would settle this, or '
            'empty string if none is needed"\n'
            "}\n"
            "If you are not sure, say so in confidence rather than inventing "
            "specifics. An honest 'uncertain' plus an experiment is worth more "
            "than a confident guess."
        )
        findings.append({
            "question": question,
            "answer": answer.get("answer", ""),
            "confidence": answer.get("confidence", "uncertain"),
            "implications": answer.get("implications", []),
            "alternatives": answer.get("alternatives", []),
            "recommendation": answer.get("recommendation", ""),
            "experiment": answer.get("experiment", ""),
        })

    synthesis = researcher.ask_json(
        f"{context}\n\nTOPIC: {topic}\n\nFINDINGS:\n{json.dumps(findings, indent=2)[:12000]}\n\n"
        "Challenge these findings, then convert them into action. Where two "
        "findings conflict, say which one wins and why. Return JSON:\n"
        "{\n"
        '  "conclusions": ["what we now believe, stated as a decision"],\n'
        '  "decisions": [{"title": "short decision title", "decision": "what we '
        'will do", "rationale": "why", "alternatives": "what we rejected and why", '
        '"consequences": "what this commits us to"}],\n'
        '  "hypotheses": [{"statement": "testable claim", "test": "how to test it"}],\n'
        '  "tickets": [{"title": "", "objective": "", "kind": "research|experiment|'
        'implementation|test|optimization", "acceptance": [""], "validation": [""]}],\n'
        '  "remaining_unknowns": ["what is still genuinely open"]\n'
        "}"
    )

    payload = {
        "simple_explanation": plan.get("simple_explanation", ""),
        "what_we_know": plan.get("what_we_know", []),
        "assumptions_challenged": plan.get("assumptions_to_challenge", []),
        "findings": findings,
        **{k: synthesis.get(k, []) for k in
           ("conclusions", "decisions", "hypotheses", "tickets", "remaining_unknowns")},
    }
    record.add(topic, payload)
    _write_notes(root, topic, payload)
    log.info("research.done", topic=topic[:60], findings=len(findings),
             tickets=len(payload.get("tickets", [])))
    return payload


def _write_notes(root: Path, topic: str, payload: dict) -> None:
    """Raw notes under research/, the distilled version under docs/research/."""
    slug = slugify(topic)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    raw = Path(root) / "research" / f"{stamp}-{slug}.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(payload, indent=2, default=str))

    def block(title: str, items, render) -> str:
        if not items:
            return ""
        return f"## {title}\n" + "\n".join(render(i) for i in items) + "\n\n"

    body = (
        f"# Research: {topic}\n\n_{time.strftime('%Y-%m-%d %H:%M')}_\n\n"
        f"## Problem in plain language\n{payload.get('simple_explanation', '')}\n\n"
        + block("What we already knew", payload.get("what_we_know"), lambda i: f"- {i}")
        + block("Findings", payload.get("findings"), lambda f:
                f"### {f['question']}\n"
                f"**Confidence:** {f['confidence']}\n\n{f['answer']}\n\n"
                + ("**Implications:**\n" + "\n".join(f"- {i}" for i in f.get("implications", []))
                   + "\n" if f.get("implications") else "")
                + (f"\n**Recommendation:** {f['recommendation']}\n" if f.get("recommendation") else ""))
        + block("Conclusions", payload.get("conclusions"), lambda i: f"- {i}")
        + block("Hypotheses", payload.get("hypotheses"), lambda h:
                f"- {h.get('statement', h)} — _test:_ {h.get('test', 'n/a')}"
                if isinstance(h, dict) else f"- {h}")
        + block("Still open", payload.get("remaining_unknowns"), lambda i: f"- {i}")
    )
    out = Path(root) / "docs" / "research" / f"{stamp}-{slug}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)


def topics_for(agents, context: str, idea: str, spec: dict, limit: int = 3,
               log: Logger = NULL) -> list[str]:
    """Decide what actually needs researching, or nothing if the path is clear."""
    result = agents["researcher"].ask_json(
        f"{context}\n\nPROJECT IDEA: {idea}\n"
        f"UNKNOWNS RECORDED AT INTAKE: {json.dumps(spec.get('unknowns', []))}\n\n"
        "Decide what genuinely requires research before implementation can "
        "start. Research is warranted only when the problem is poorly "
        "understood, several approaches compete, the best method is uncertain, "
        "or the reasoning is non-obvious. Routine engineering needs no research."
        f"\n\nReturn JSON: {{\"needed\": true|false, \"topics\": [\"specific "
        f"research question\"]}}. At most {limit} topics. Return needed=false "
        "with an empty list when the implementation path is already clear."
    )
    if not result.get("needed"):
        log.info("research.skipped", reason="path already clear")
        return []
    return [t for t in (result.get("topics") or [])[:limit] if isinstance(t, str)]
