"""Feynman-style research workflow.

The loop is: explain the problem simply, name the unknowns, attack them,
challenge the resulting assumptions, and convert what survives into
decisions, hypotheses and tickets. Output must change what gets built --
a summary that changes nothing is treated as a failed round.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agents import Prompt
from .log import Logger, NULL
from .sources import as_records
from .workspace import slugify


@dataclass
class Unknown:
    id: str
    question: str
    why_it_matters: str = ""
    resolution: str = ""
    confidence: str = "unknown"      # unknown | inferred | verified
    needs_experiment: bool = False


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

    def known(self, budget: int = 8000) -> str:
        """Everything established so far, for feeding back into later rounds.

        Newest first into the budget, oldest dropped: a finding from round one
        has usually already become a decision in the spec, while the one from
        the round before this is what this topic follows from.
        """
        lines = []
        for round_ in self.rounds:
            for finding in round_.get("findings", []):
                lines.append(f"- ({finding.get('confidence', '?')}) "
                             f"{finding.get('question')}: {finding.get('answer')}")
        kept: list[str] = []
        used = 0
        for line in reversed(lines[-40:]):
            if kept and used + len(line) + 1 > budget:
                break
            used += len(line) + 1
            kept.append(line)
        return "\n".join(reversed(kept)) or "(nothing established yet)"


def investigate(agents, root: Path, topic: str, context: str, *,
                max_unknowns: int = 6, log: Logger = NULL, librarian=None) -> dict:
    """Run one Feynman round on *topic*. Returns findings and actions.

    With a *librarian* the round first retrieves real sources and answers
    from them; without one it answers from what the model already knows,
    which is why every finding carries an explicit confidence either way.
    """
    researcher = agents["researcher"]
    record = ResearchLog(root)
    log.info("research.round", topic=topic[:80])

    plan_prompt = Prompt(researcher, log=log)
    plan_prompt.add("context", context, priority=20, floor=600)
    plan_prompt.add("topic", f"RESEARCH TOPIC: {topic}", required=True)
    # What earlier rounds established is what stops this one re-asking their
    # questions, so it outranks the project context; the tail is kept because
    # the latest rounds are the ones this topic follows from.
    established = plan_prompt.add("established", priority=10, floor=800, tail=True)
    plan_prompt.add(
        "instructions",
        "Step 1 of the Feynman method. Return JSON:\n"
        "{\n"
        '  "simple_explanation": "explain the problem as if to a smart person '
        'outside the field, in 3-5 sentences",\n'
        '  "what_we_know": ["established facts that need no research"],\n'
        '  "unknowns": [{"question": "specific answerable question", '
        '"why_it_matters": "what decision it changes", '
        '"needs_experiment": true|false}],\n'
        '  "assumptions_to_challenge": ["assumption that could be wrong"]\n'
        "}\n"
        f"List at most {max_unknowns} unknowns, ordered by how much each would "
        "change the implementation. Skip anything already established.",
        required=True)
    header = "ALREADY ESTABLISHED:\n"
    established.fill(header + record.known(max(0, established.room() - len(header))))
    plan = researcher.ask_json(plan_prompt.render())
    unknowns = (plan.get("unknowns") or [])[:max_unknowns]

    refused_from = len(librarian.refusals) if librarian else 0
    sources, material = [], ""
    if librarian is not None and unknowns:
        sources = _consult(researcher, librarian, topic, unknowns, context, log)
        material = librarian.excerpts(sources)

    findings = []
    for index, unknown in enumerate(unknowns, 1):
        question = unknown.get("question") if isinstance(unknown, dict) else str(unknown)
        if not question:
            continue
        log.info("research.unknown", n=index, question=question[:80])
        ask = Prompt(researcher, log=log)
        ask.add("context", context, priority=20, floor=600)
        ask.add("question", f"QUESTION: {question}\nWHY IT MATTERS: "
                + (unknown.get("why_it_matters", "") if isinstance(unknown, dict) else ""),
                required=True)
        # Retrieved text is the only thing here the model does not already
        # have, so it outranks the project context it was fetched to answer.
        ask.add("material",
                ("RETRIEVED SOURCES — this text was fetched for this round. "
                 "Prefer it over memory, and cite the ones you actually used:\n"
                 f"{material}" if material else ""), priority=10, floor=1500)
        ask.add("instructions",
            ("Answer from the retrieved sources where they cover the question, "
             "and from what you know where they do not. Say which is which.\n"
             if material else "Answer from what you actually know.\n")
            + "Return JSON:\n"
            "{\n"
            '  "answer": "the substantive answer, with concrete names, algorithms, '
            'complexities, formats or numbers where they exist",\n'
            '  "confidence": "verified|inferred|uncertain",\n'
            + ('  "sources": ["url or [n] of a retrieved source you actually used"],\n'
               if material else "")
            + '  "implications": ["what this means for the implementation"],\n'
            '  "alternatives": [{"option": "name", "pros": "", "cons": ""}],\n'
            '  "recommendation": "which alternative to take and why",\n'
            '  "experiment": "a measurable experiment that would settle this, or '
            'empty string if none is needed"\n'
            "}\n"
            "If you are not sure, say so in confidence rather than inventing "
            "specifics. An honest 'uncertain' plus an experiment is worth more "
            "than a confident guess.", required=True)
        answer = researcher.ask_json(ask.render())
        findings.append({
            "question": question,
            "answer": answer.get("answer", ""),
            "confidence": answer.get("confidence", "uncertain"),
            "sources": _cited(answer.get("sources"), sources),
            "implications": answer.get("implications", []),
            "alternatives": answer.get("alternatives", []),
            "recommendation": answer.get("recommendation", ""),
            "experiment": answer.get("experiment", ""),
        })

    close = Prompt(researcher, log=log)
    close.add("context", context, priority=20, floor=600)
    close.add("topic", f"TOPIC: {topic}", required=True)
    answers = close.add("findings", priority=10, floor=2000)
    close.add("instructions",
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
        "}", required=True)
    header = "FINDINGS:\n"
    answers.fill(header + json.dumps(findings, indent=2)[
        : max(0, answers.room() - len(header))])
    synthesis = researcher.ask_json(close.render())

    payload = {
        "simple_explanation": plan.get("simple_explanation", ""),
        "what_we_know": plan.get("what_we_know", []),
        "assumptions_challenged": plan.get("assumptions_to_challenge", []),
        "findings": findings,
        "sources": as_records(sources),
        "unreachable": librarian.refusals[refused_from:] if librarian else [],
        **{k: synthesis.get(k, []) for k in
           ("conclusions", "decisions", "hypotheses", "tickets", "remaining_unknowns")},
    }
    record.add(topic, payload)
    _write_notes(root, topic, payload)
    log.info("research.done", topic=topic[:60], findings=len(findings),
             sources=len(sources), tickets=len(payload.get("tickets", [])))
    return payload


def _consult(researcher, librarian, topic: str, unknowns: list, context: str,
             log: Logger) -> list:
    """Ask what is worth reading, then have the sandbox fetch it."""
    questions = "\n".join(
        f"- {u.get('question') if isinstance(u, dict) else u}" for u in unknowns)
    wanted = researcher.ask_json(
        f"{context}\n\nTOPIC: {topic}\n\nQUESTIONS TO ANSWER:\n{questions}\n\n"
        f"Name at most {librarian.remaining} pages worth retrieving to answer "
        "these. Prefer primary sources: official documentation, a specification "
        "or RFC, a project's own repository or reference manual, a paper. Give a "
        "URL only if you are confident it exists and is publicly reachable — an "
        "invented URL spends part of the budget and returns nothing. Return JSON:\n"
        '{"sources": [{"url": "https://...", "why": "what it should settle"}]}'
    )
    found = librarian.gather((wanted.get("sources") or [])[:librarian.remaining])
    log.info("research.consulted", topic=topic[:60], retrieved=len(found),
             refused=len(librarian.refusals))
    return found


def _cited(claimed, sources: list) -> list[str]:
    """Keep only citations naming something that was actually retrieved.

    A model asked for citations supplies plausible ones whether or not it
    read anything, which is the failure networked research exists to remove.
    Excerpts are numbered in the prompt, so ``[2]`` is accepted as a
    reference to the second source and resolved to its URL.
    """
    urls = {s.url for s in sources}
    kept = []
    for claim in claimed or []:
        value = str(claim.get("url", "") if isinstance(claim, dict) else claim).strip()
        index = value.strip("[]")
        if index.isdigit() and 1 <= int(index) <= len(sources):
            value = sources[int(index) - 1].url
        if value in urls and value not in kept:
            kept.append(value)
    return kept


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
                + ("**Sources:** " + ", ".join(f"<{u}>" for u in f["sources"]) + "\n\n"
                   if f.get("sources") else "")
                + ("**Implications:**\n" + "\n".join(f"- {i}" for i in f.get("implications", []))
                   + "\n" if f.get("implications") else "")
                + (f"\n**Recommendation:** {f['recommendation']}\n" if f.get("recommendation") else ""))
        + block("Conclusions", payload.get("conclusions"), lambda i: f"- {i}")
        + block("Hypotheses", payload.get("hypotheses"), lambda h:
                f"- {h.get('statement', h)} — _test:_ {h.get('test', 'n/a')}"
                if isinstance(h, dict) else f"- {h}")
        + block("Still open", payload.get("remaining_unknowns"), lambda i: f"- {i}")
        + block("Sources retrieved", payload.get("sources"), lambda s:
                f"- [{s.get('title') or s.get('url')}]({s.get('url')}) — "
                f"retrieved copy: `{s.get('path')}`")
        + block("Could not be retrieved", payload.get("unreachable"), lambda r:
                f"- {r.get('url')} — {r.get('reason')}")
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
