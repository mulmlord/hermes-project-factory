"""Feynman-style research workflow.

The loop is: explain the problem simply, name the unknowns, attack them,
challenge the resulting assumptions, and convert what survives into
decisions, hypotheses and tickets. Output must change what gets built --
a summary that changes nothing is treated as a failed round.

When the round reads real pages it is also the one place where text the
factory did not write reaches an agent, so the round is built in three
layers and only the first of them ever sees a page.

**Quarantine.** :func:`_read` asks one question of the retrieved material on
a call with no project context and no toolbox -- :class:`hermes.tools.Lookup`
refuses one to any prompt carrying an untrusted block -- and its answer is
reduced in Python to at most a handful of short claims, each pinned to a
source that was really fetched. A claim whose citation does not resolve is
dropped, so untrusted text can only travel attached to evidence.

**Privilege.** Every later prompt -- the answer, the synthesis, the
architect reading the summary -- sees the claims, never the page. That is
the Dual-LLM split, and its published weakness is exactly the thing the
third layer is for: it constrains what may flow, not what the flowing
values say.

**Policy.** :func:`_screen` decides in Python what a round that read pages
is allowed to have proposed. An install is refused outright and every other
command must clear the project's own security gate and the classifier. None
of this is expressed as an instruction to a model, because an instruction to
a model is precisely what the page is competing with.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import prompts
from .agents import Prompt
from .contextdb import UNTRUSTED
from .log import Logger, NULL
from .sources import as_records, flatten, quarantine
from .verify import installs, text_gate
from .workspace import slugify

#: Characters one claim may carry out of the quarantine, and how many claims
#: one question may take out of it. A claim is a sentence the page supports;
#: a page quoted whole is the page again, and not travelling whole is the
#: entire point of reading it behind a wall.
CLAIM_CHARS = 400
MAX_CLAIMS = 6


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
                max_unknowns: int = 6, log: Logger = NULL, librarian=None,
                db=None, classifier=None) -> dict:
    """Run one Feynman round on *topic*. Returns findings and actions.

    With a *librarian* the round first retrieves real sources and answers
    from them; without one it answers from what the model already knows,
    which is why every finding carries an explicit confidence either way.

    *db* is the context database, where each retrieved page is recorded as
    an untrusted source so the next agent can see what this round read and
    where it came from. *classifier* is the security classifier, which joins
    the gate in screening what a round that read pages proposed.
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
    plan_prompt.add("instructions", prompts.feynman_unknowns(max_unknowns),
                    required=True)
    header = "ALREADY ESTABLISHED:\n"
    established.fill(header + record.known(max(0, established.room() - len(header))))
    plan = researcher.ask_json(plan_prompt.render())
    unknowns = (plan.get("unknowns") or [])[:max_unknowns]

    refused_from = len(librarian.refusals) if librarian else 0
    sources = []
    if librarian is not None:
        librarian.start_round()      # the run's page budget does not reset
    if librarian is not None and unknowns:
        sources = _consult(researcher, librarian, topic, unknowns, context, log)
        _remember_sources(db, sources, topic)

    findings = []
    for index, unknown in enumerate(unknowns, 1):
        question = unknown.get("question") if isinstance(unknown, dict) else str(unknown)
        if not question:
            continue
        log.info("research.unknown", n=index, question=question[:80])
        why = unknown.get("why_it_matters", "") if isinstance(unknown, dict) else ""
        claims = _read(researcher, librarian, sources, question, log) if sources else []
        ask = Prompt(researcher, log=log)
        ask.add("context", context, priority=20, floor=600)
        ask.add("question", f"QUESTION: {question}\nWHY IT MATTERS: {why}",
                required=True)
        # What the pages said, as claims rather than as pages: this prompt has
        # the project in it, and nothing off the network reaches it unquoted.
        ask.add("material", prompts.retrieved_material(_quoted(librarian, claims)),
                priority=10, floor=1500, trust=UNTRUSTED)
        ask.add("instructions", prompts.feynman_answer(bool(claims)),
                required=True)
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
    close.add("instructions", prompts.feynman_synthesis(), required=True)
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
    # Only a round that actually read pages is screened. One answering from
    # the model's own weights proposes what the rest of the factory proposes,
    # and refusing its setup command would be a regression, not a defence.
    payload["refused"] = _screen(payload, classifier=classifier,
                                 log=log) if sources else []
    record.add(topic, payload)
    _write_notes(root, topic, payload)
    log.info("research.done", topic=topic[:60], findings=len(findings),
             sources=len(sources), tickets=len(payload.get("tickets", [])))
    return payload


def _read(researcher, librarian, sources: list, question: str,
          log: Logger) -> list[dict]:
    """Ask the retrieved pages one question, behind the wall.

    The whole of what this call is given is the question and the fenced
    material: no project context, no architecture, no build commands, and no
    toolbox -- a prompt with an untrusted block cannot be handed one. So a
    page that talks the model into cooperating has bought an agent that
    cannot read a file, cannot run anything, and can only return claims.

    The material is sized to the room the block actually has before it is
    rendered, rather than trimmed afterwards, because a trim cuts the tail
    and the tail is the closing marker.
    """
    prompt = Prompt(researcher, log=log)
    prompt.add("question", f"QUESTION: {question}", required=True)
    body = prompt.add("material", priority=10, floor=1500, trust=UNTRUSTED)
    prompt.add("instructions", prompts.quarantined_reading(MAX_CLAIMS),
               required=True)
    body.fill(prompts.retrieved_material(
        librarian.excerpts(sources, budget=max(1000, body.room() - 1000))))
    claims = _claims(researcher.ask_json(prompt.render()), sources)
    log.info("research.quarantined_read", question=question[:80],
             claims=len(claims), sources=len(sources))
    return claims


def _claims(answer: Any, sources: list) -> list[dict]:
    """What may leave the quarantine: short claims that name a real source.

    Three rules, all enforced here rather than asked for in the prompt. A
    claim is flattened to one line, so it cannot forge a heading or a fence
    in whatever prompt quotes it next. It is cut to :data:`CLAIM_CHARS`, so
    a page cannot escape by being quoted whole. And it is dropped unless its
    citation resolves to a page this round really fetched, so untrusted text
    only travels with the evidence for it attached.
    """
    if not isinstance(answer, dict):
        return []
    kept: list[dict] = []
    for item in (answer.get("claims") or []):
        if not isinstance(item, dict):
            continue
        claim = flatten(item.get("claim", ""), CLAIM_CHARS)
        cited = _cited([item.get("source")], sources)
        if not claim or not cited:
            continue
        kept.append({"claim": claim, "source": cited[0]})
        if len(kept) >= MAX_CLAIMS:
            break
    return kept


def _quoted(librarian, claims: list[dict]) -> str:
    """The claims as a fenced block, for the prompts that do hold privilege."""
    if not claims:
        return ""
    return quarantine(librarian.mark,
                      [(f"<{c['source']}>", c["claim"]) for c in claims])


def _remember_sources(db, sources: list, topic: str) -> None:
    """Record what this round read, marked untrusted, in the context database.

    Metadata only -- the citation, why it was wanted, where the bytes were
    kept. Not the text: :meth:`hermes.contextdb.ContextDB.search` is what the
    ``recall`` tool reads, and that tool is offered to agents that *do* hold
    privilege, so putting page text there would rebuild the channel this
    module exists to close. The title still comes off the page, which is why
    the record is untrusted even though it is three lines long.
    """
    if db is None:
        return
    for source in sources:
        db.record("source", flatten(source.cite(), 300),
                  f"retrieved for research on {topic}\n"
                  f"wanted for: {flatten(source.why, 200) or '(not stated)'}\n"
                  f"copy kept at: {source.path}",
                  agent="researcher", paths=[source.path],
                  trust=source.trust, weight=4,
                  meta={"url": source.url, "bytes": source.bytes})


def _screen(payload: dict, *, classifier=None, log: Logger = NULL) -> list[dict]:
    """Refuse the executable actions a retrieved page talked the round into.

    A round that read pages may still propose work -- learning something is
    the point -- but the parts of that work the factory would *run* do not
    get to arrive unexamined, and no wording in the prompt above can be the
    thing that examines them.

    Two rules. An install is refused outright: nothing a stranger's page
    says is a reason to add a package to the sandbox, and research is not
    where dependencies are chosen -- the architect picks those from the
    specification and the setup command is where they belong. Everything
    else must clear the project's own security gate and the classifier over
    the command text, with the classifier blocking rather than advising,
    because its false positive here costs one proposed command instead of a
    correct ticket.

    What is refused is returned rather than dropped silently: a command that
    vanished without a record reads as a model that forgot to write one.
    """
    refused: list[dict] = []
    for ticket in payload.get("tickets") or []:
        if not isinstance(ticket, dict):
            continue
        raw = ticket.get("validation")
        commands = [raw] if isinstance(raw, str) else list(raw or [])
        kept = []
        for command in commands:
            command = str(command)
            why = _refuse_command(command, classifier=classifier)
            if why:
                refused.append({"command": flatten(command, 300),
                                "ticket": flatten(ticket.get("title", ""), 120),
                                "reason": why})
            else:
                kept.append(command)
        ticket["validation"] = kept
    if refused:
        log.warn("research.action_refused", count=len(refused),
                 first=refused[0]["reason"][:200])
    return refused


def _refuse_command(command: str, *, classifier=None) -> str:
    """Why this command may not come out of a round that read pages, or ""."""
    manager = installs(command)
    if manager:
        return (f"a research round that read untrusted pages may not install "
                f"anything ({manager})")
    gate = text_gate({"proposed command": command}, classifier=classifier,
                     block=True)
    return "" if gate.passed else gate.detail


def _consult(researcher, librarian, topic: str, unknowns: list, context: str,
             log: Logger) -> list:
    """Turn the questions into URLs that exist, then have the sandbox fetch them.

    The model writes search queries rather than URLs. Asking a local model to
    recall a URL is asking it to invent one, and an invented URL spends a page
    of the budget to return a 404; choosing among results a search engine
    returned is a far easier job, and a wrong choice costs a page rather than
    producing a citation to a page that never existed. A URL the model says it
    is certain of is still tried, but only after the results.
    """
    if not librarian.remaining or not librarian.searches_left:
        # Without a search left there is nothing to pick from, and a round that
        # falls back to recalled URLs spends pages to answer from memory anyway.
        log.info("research.budget_spent", pages=librarian.spent,
                 searches=librarian.searched)
        return []
    questions = "\n".join(
        f"- {u.get('question') if isinstance(u, dict) else u}" for u in unknowns)
    queries = min(3, librarian.searches_left)
    plan = researcher.ask_json(
        prompts.search_queries(context, topic, questions, queries))
    candidates: list[dict] = []
    seen: set[str] = set()
    for query in [q for q in (plan.get("queries") or [])
                  if isinstance(q, str) and q.strip()][:queries]:
        for hit in librarian.search(query):
            if hit["url"] not in seen:
                seen.add(hit["url"])
                candidates.append(hit)
    wanted = _choose(researcher, candidates, topic, questions,
                     librarian.remaining, log)
    wanted += [u for u in (plan.get("urls") or [])
               if isinstance(u, dict) and u.get("url")]
    found = librarian.gather(wanted[:librarian.remaining])
    log.info("research.consulted", topic=topic[:60], candidates=len(candidates),
             retrieved=len(found), refused=len(librarian.refusals))
    return found


def _choose(researcher, candidates: list, topic: str, questions: str, limit: int,
            log: Logger) -> list[dict]:
    """Pick which search results are worth spending a page of the budget on."""
    if not candidates or limit <= 0:
        return []
    listing = "\n".join(f"[{n}] {hit['title'] or '(no title)'}\n    {hit['url']}"
                        for n, hit in enumerate(candidates, 1))
    picked = researcher.ask_json(
        prompts.choose_sources(topic, questions, listing, limit))
    wanted: list[dict] = []
    for item in (picked.get("pick") or []):
        number = item.get("n") if isinstance(item, dict) else item
        try:
            index = int(str(number).strip("[]"))
        except (TypeError, ValueError):
            continue
        if not 1 <= index <= len(candidates):
            continue
        hit = candidates[index - 1]
        why = item.get("why", "") if isinstance(item, dict) else ""
        if hit["url"] not in {w["url"] for w in wanted}:
            wanted.append({"url": hit["url"], "why": why or hit["title"]})
    if not wanted:
        # An unusable answer should not throw the search away: the engine's
        # own ranking is a better default than falling back to memory.
        log.info("research.pick_unusable", candidates=len(candidates))
        wanted = [{"url": hit["url"], "why": hit["title"]}
                  for hit in candidates[:limit]]
    return wanted[:limit]


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
        # A refusal is written down where a person will see it: a command that
        # disappeared silently looks like a model that forgot to propose one,
        # and the difference matters when a page was trying to place it.
        + block("Refused after reading untrusted pages", payload.get("refused"),
                lambda r: f"- `{r.get('command')}` "
                          f"({r.get('ticket') or 'unnamed ticket'}) — {r.get('reason')}")
    )
    out = Path(root) / "docs" / "research" / f"{stamp}-{slug}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)


def topics_for(agents, context: str, idea: str, spec: dict, limit: int = 3,
               log: Logger = NULL) -> list[str]:
    """Decide what actually needs researching, or nothing if the path is clear."""
    result = agents["researcher"].ask_json(prompts.research_topics(
        context, idea, json.dumps(spec.get("unknowns", [])), limit))
    if not result.get("needed"):
        log.info("research.skipped", reason="path already clear")
        return []
    return [t for t in (result.get("topics") or [])[:limit] if isinstance(t, str)]
