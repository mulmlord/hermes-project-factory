"""Every task prompt the factory sends, in one place.

:mod:`hermes.agents` holds the eleven system prompts -- who each role is.
This module holds what each role is asked to *do*: the task statements, the
output schemas and the rules attached to them. Together they are the whole of
Hermes' prompting, and :data:`hermes.improve.TARGETS` points the ``prompts``
self-improvement target at both, so a proposal to reword a prompt is a diff of
prompts rather than a diff of control flow.

What lives here is instruction: anything that tells a model what to produce.
What does not is data -- a file listing, a ticket rendered for reading, a
failure log -- which stays at its call site, where the budget that sizes it
is. A caller assembles blocks; this module supplies the words.

The prompt that proposes self-modifications is the one exception, and it
stays in :mod:`hermes.improve` on purpose: that module is protected from the
candidates it evaluates, and the instruction soliciting a candidate belongs
on the protected side of that line.
"""
from __future__ import annotations

from .agents import EDIT_RULES

# -- intake and planning -----------------------------------------------------


def spec_from_idea(idea: str) -> str:
    """Turn a one-line idea into a formal specification."""
    return (
        f"PROJECT IDEA (the only input the user gave):\n{idea}\n\n"
        "Produce a complete specification. Infer sensible defaults for anything "
        "unstated and record each inference under \"assumptions\" -- you cannot "
        "ask the user. Return JSON:\n"
        "{\n"
        '  "name": "short-kebab-case-project-name",\n'
        '  "objective": "one paragraph: what this software does and for whom",\n'
        '  "deliverables": ["concrete artifact the project must produce"],\n'
        '  "platforms": ["target OS/runtime"],\n'
        '  "technologies": ["language and key libraries, with the language first"],\n'
        '  "functional_requirements": ["testable statement of required behaviour"],\n'
        '  "nonfunctional_requirements": ["performance, security, reliability"],\n'
        '  "constraints": ["hard limits the implementation must respect"],\n'
        '  "risks": ["what could make this fail"],\n'
        '  "unknowns": ["what is not yet known and must be researched"],\n'
        '  "success_criteria": ["measurable, checkable completion condition"],\n'
        '  "assumptions": ["defaults you chose because the idea did not say"],\n'
        '  "optimization_objective": {"metric": "name of an automatically '
        'measurable metric, or empty string if none applies", "direction": '
        '"minimize|maximize", "how_measured": "how a script would measure it"}\n'
        "}\n"
        "Scope it so a small team could build it: ambitious but finishable. "
        "Prefer the standard library and few dependencies."
    )


def system_design(spec_json: str, research: str) -> str:
    """Choose the architecture and the commands that build and check it."""
    return (
        f"SPECIFICATION:\n{spec_json}\n\n"
        f"RESEARCH CONCLUSIONS:\n{research or '(none)'}\n\n"
        "Design the system and define how it is built and verified. Return JSON:\n"
        "{\n"
        '  "approach": "how you will build this, in a short paragraph",\n'
        '  "architecture": "components and how they fit together",\n'
        '  "components": [{"name": "", "responsibility": "", "path": "src/..."}],\n'
        '  "milestones": [{"name": "", "description": "", "exit_criteria": ""}],\n'
        '  "commands": {"setup": "shell command installing dependencies", '
        '"build": "shell command that compiles/checks, or \'true\' for '
        'interpreted projects", "test": "shell command running the test suite", '
        '"lint": "shell command or empty", "benchmark": "shell command printing '
        'a score, or empty"},\n'
        '  "layout": ["files the project will contain, with one-line purpose"],\n'
        '  "decisions": [{"title": "", "decision": "", "rationale": "", '
        '"alternatives": "", "consequences": ""}]\n'
        "}\n"
        "Commands must be real, runnable, non-interactive shell commands that "
        "work from the project root inside a Linux container. Never use sudo. "
        "The container is a bare base image with nothing but the language "
        "runtime installed. Only the setup command has network access, so it "
        "must install every tool the build, test and lint commands invoke "
        "(for example 'pip install pytest flake8'). Every other command runs "
        "offline."
    )


def ticket_breakdown(spec_json: str, plan_json: str, max_tickets: int,
                     kinds: str) -> str:
    """Decompose the plan into an ordered, dependency-aware ticket set."""
    return (
        f"SPECIFICATION:\n{spec_json}\n\n"
        f"PLAN:\n{plan_json}\n\n"
        f"Break this into at most {max_tickets} tickets. Each ticket must be "
        "small enough for one agent to implement, test and verify in a single "
        "pass -- roughly one component or one coherent feature. Order them so "
        "foundations come first. Return JSON:\n"
        "{\n"
        '  "tickets": [{\n'
        '    "key": "short-unique-key",\n'
        '    "title": "imperative summary",\n'
        '    "objective": "what done looks like, concretely",\n'
        '    "context": "what the implementer needs to know",\n'
        f'    "kind": "one of {kinds}",\n'
        '    "dependencies": ["key of a ticket that must finish first"],\n'
        '    "acceptance": ["checkable criterion"],\n'
        '    "validation": ["shell command proving it works"],\n'
        '    "artifacts": ["file this ticket creates or changes"],\n'
        '    "priority": 1, "risk": 3, "cost": 2, "info_gain": 2\n'
        "  }]\n"
        "}\n"
        "priority/risk/cost/info_gain are 1-5. Include tickets for tests and "
        "benchmarks where they apply. Do NOT include a README ticket; "
        "documentation is generated at the end. Dependencies must reference "
        "keys in this same list and must not form a cycle."
    )


def ticket_split(ticket_json: str, reason: str) -> str:
    """Split a ticket that proved too large into executable children."""
    return (
        f"This ticket is too large or keeps failing and must be split.\n\n"
        f"TICKET: {ticket_json}\n\n"
        f"WHY IT IS BEING SPLIT:\n{reason}\n\n"
        "Return JSON with 2-5 smaller tickets that together fully replace it:\n"
        '{"tickets": [{"key": "", "title": "", "objective": "", "context": "", '
        '"kind": "", "dependencies": ["key of an earlier ticket in this list"], '
        '"acceptance": [""], "validation": [""], "artifacts": [""], '
        '"priority": 1, "risk": 2, "cost": 2, "info_gain": 2}]}\n'
        "Each child must be strictly smaller and independently verifiable."
    )


def repair_tickets(context: str, report: str, output: str) -> str:
    """Turn a failing project validation into repair tickets."""
    return (
        f"{context}\n\nPROJECT VALIDATION FAILED:\n{report}\n\n"
        f"OUTPUT:\n{output}\n\n"
        "Create repair tickets. Return JSON:\n"
        '{"tickets": [{"key": "", "title": "", "objective": "", "context": "", '
        '"kind": "implementation", "acceptance": [""], "validation": [""], '
        '"artifacts": [""], "priority": 1, "risk": 2, "cost": 2, "info_gain": 1}]}'
    )


# -- research ----------------------------------------------------------------


def research_topics(context: str, idea: str, unknowns_json: str,
                    limit: int) -> str:
    """Decide what actually needs researching, or nothing if the path is clear."""
    return (
        f"{context}\n\nPROJECT IDEA: {idea}\n"
        f"UNKNOWNS RECORDED AT INTAKE: {unknowns_json}\n\n"
        "Decide what genuinely requires research before implementation can "
        "start. Research is warranted only when the problem is poorly "
        "understood, several approaches compete, the best method is uncertain, "
        "or the reasoning is non-obvious. Routine engineering needs no research."
        f"\n\nReturn JSON: {{\"needed\": true|false, \"topics\": [\"specific "
        f"research question\"]}}. At most {limit} topics. Return needed=false "
        "with an empty list when the implementation path is already clear."
    )


def feynman_unknowns(max_unknowns: int) -> str:
    """Step 1: explain the problem simply and name what is genuinely unknown."""
    return (
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
        "change the implementation. Skip anything already established."
    )


def retrieved_material(material: str) -> str:
    """Frame fetched text so it is preferred over what the model recalls."""
    if not material:
        return ""
    return ("RETRIEVED SOURCES — this text was fetched for this round. "
            "Prefer it over memory, and cite the ones you actually used:\n"
            f"{material}")


def feynman_answer(has_material: bool) -> str:
    """Answer one unknown, from the sources where they cover it."""
    return (
        ("Answer from the retrieved sources where they cover the question, "
         "and from what you know where they do not. Say which is which.\n"
         if has_material else "Answer from what you actually know.\n")
        + "Return JSON:\n"
        "{\n"
        '  "answer": "the substantive answer, with concrete names, algorithms, '
        'complexities, formats or numbers where they exist",\n'
        '  "confidence": "verified|inferred|uncertain",\n'
        + ('  "sources": ["url or [n] of a retrieved source you actually used"],\n'
           if has_material else "")
        + '  "implications": ["what this means for the implementation"],\n'
        '  "alternatives": [{"option": "name", "pros": "", "cons": ""}],\n'
        '  "recommendation": "which alternative to take and why",\n'
        '  "experiment": "a measurable experiment that would settle this, or '
        'empty string if none is needed"\n'
        "}\n"
        "If you are not sure, say so in confidence rather than inventing "
        "specifics. An honest 'uncertain' plus an experiment is worth more "
        "than a confident guess."
    )


def feynman_synthesis() -> str:
    """Challenge the round's findings and convert what survives into action."""
    return (
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


def search_queries(context: str, topic: str, questions: str, queries: int) -> str:
    """Ask for search terms rather than recalled URLs."""
    return (
        f"{context}\n\nTOPIC: {topic}\n\nQUESTIONS TO ANSWER:\n{questions}\n\n"
        f"Write at most {queries} web search queries that would find primary "
        "sources for these questions: official documentation, a specification "
        "or RFC, a project's own repository or reference manual, a paper. Write "
        "search queries, not questions — the terms an expert would type, "
        "including the exact names of the technologies involved. If you are "
        "certain a specific URL exists and is publicly reachable, you may list "
        "it as well. Return JSON:\n"
        '{"queries": ["search terms"], '
        '"urls": [{"url": "https://...", "why": "what it should settle"}]}'
    )


def choose_sources(topic: str, questions: str, listing: str, limit: int) -> str:
    """Pick which search results are worth spending a page of the budget on."""
    return (
        f"TOPIC: {topic}\n\nQUESTIONS TO ANSWER:\n{questions}\n\n"
        f"SEARCH RESULTS:\n{listing}\n\n"
        f"Choose at most {limit} of these results worth reading in full, best "
        "first. Prefer primary sources over blog posts and aggregators, and "
        "skip anything that only restates what another result covers. Refer to "
        "each by its number. Return JSON:\n"
        '{"pick": [{"n": 1, "why": "what it should settle"}]}'
    )


def compress_findings(words: int) -> str:
    """Summarise research that no longer fits, without losing a constraint."""
    return (
        f"Compress this to at most {words} words for an architect who "
        "must choose a design from it. Keep every decision, constraint, "
        "measured number, named technology and rejected alternative. "
        'Return JSON: {"findings": ["one finding per entry"]}'
    )


# -- implementation ----------------------------------------------------------

#: Said to a stronger role on a repeated failure. Swapping the role without
#: saying why buys little: the failure text is the same one the last attempt
#: read, and it argues for the same small patch.
ESCALATION = (
    "EARLIER ATTEMPTS AT THIS TICKET FAILED. Do not resend the last "
    "approach with small edits: work out why it cannot be made to "
    "work, decide how the ticket should be implemented instead, and "
    "write that."
)

#: The closing line of every implementation prompt.
IMPLEMENT = (
    "Implement this ticket completely. Every file you return "
    "must be complete and runnable. If the ticket needs tests, write "
    "real ones."
)


# -- review and triage -------------------------------------------------------

REVIEW = (
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

TRIAGE = (
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


# -- experiments -------------------------------------------------------------


def experiment_design(timeout: int, protocol: str) -> str:
    """Ask for the smallest program that settles a claim by measuring it."""
    return (
        "Build the smallest program that settles this hypothesis by "
        "measuring it. Requirements:\n"
        "- self-contained: every file lives in the experiment directory, "
        "and your command is run from inside that directory\n"
        "- no network: the sandbox has none\n"
        "- only the standard library and what the project already depends on\n"
        "- deterministic: fixed seeds, fixed inputs, so two runs agree\n"
        f"- finishes well inside {timeout} seconds\n"
        "- measures the claim rather than asserting it. A program that "
        "prints what the hypothesis says, without computing it, settles "
        "nothing and is worse than no experiment.\n\n"
        + protocol + "\n\nReturn JSON:\n"
        "{\n"
        '  "design": "what is measured, and why that settles the hypothesis",\n'
        '  "parameters": {"name": value},\n'
        '  "files": [{"path": "run.py", "content": "<COMPLETE file contents>"}],\n'
        '  "command": "python3 run.py",\n'
        '  "supports": "the result that would support the hypothesis",\n'
        '  "refutes": "the result that would refute it"\n'
        "}"
    )


EXPERIMENT_VERDICT = (
    "Judge the hypothesis against this measurement and nothing else. "
    "If the program does not actually measure the claim, the verdict "
    "is inconclusive and the finding says why. A refuted hypothesis is "
    "a good result; do not soften it.\nReturn JSON:\n"
    "{\n"
    '  "verdict": "supported|refuted|inconclusive",\n'
    '  "finding": "what this establishes, with the numbers in it",\n'
    '  "confidence": "verified|inferred|uncertain",\n'
    '  "implications": ["what this changes about what gets built"],\n'
    '  "next": "the action this argues for, or an empty string"\n'
    "}"
)


# -- evolution ---------------------------------------------------------------


def evolution_direction(family: str) -> str:
    """Ask for one focused change in this generation's direction."""
    return (
        f"THIS GENERATION'S DIRECTION: {family}\n\n"
        "Propose ONE focused change in that direction that improves the "
        "objective while keeping every test passing. Do not repeat a change "
        "listed as not helping. Return JSON:\n"
        '{"mutation": "one line naming the change", '
        '"mechanism": "why this is faster/better, mechanically", '
        '"edits": [{"path": "src/thing.ext", "find": "<exact text to '
        'replace>", "replace": "<the text that takes its place>"}], '
        '"files": [{"path": "src/thing.ext", "content": "<COMPLETE file '
        'contents>"}], '
        '"risk": "what could break"}\n' + EDIT_RULES
        + "- Change only the files shown above."
    )


def evolution_targets(context: str, benchmark) -> str:
    """Find the source files that actually dominate the benchmark's cost."""
    return (
        f"{context}\n\n"
        f"BENCHMARK COMMAND: {benchmark}\n\n"
        "Which existing source files dominate this benchmark's cost? Return "
        'JSON: {"files": ["src/path.ext"]} with at most 4 files that exist in '
        "the file listing above. Exclude tests and benchmark harnesses."
    )


# -- documentation -----------------------------------------------------------


def readme(sections: str) -> str:
    """Write the README from the source, after everything else has run."""
    return (
        "Write the project README. Document only what the source above actually "
        "does; never invent a feature or a flag. Every command must be one a user "
        "can really run from a clean checkout. Return JSON with a markdown string "
        "for each key:\n"
        "{" + sections + "}\n"
        "purpose: 2-3 sentences. features: a bulleted list of what works today. "
        "architecture: how the pieces fit, with the real file names. examples: at "
        "least one complete runnable example with its expected output. "
        "limitations: honest, including anything listed as not implemented."
    )


ARCHITECTURE_DOC = (
    "Document the architecture as built. Return JSON:\n"
    '{"overview": "", "components": [{"name": "", "path": "", '
    '"responsibility": "", "interfaces": ""}], "data_flow": "", '
    '"key_decisions": [""], "extension_points": [""]}'
)
