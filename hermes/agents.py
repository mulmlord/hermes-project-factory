"""Specialised agent roles.

Agents are stateless: everything they need is passed in from project state,
so any agent can be re-run after a crash without replaying a conversation.
Each returns plain Python data, never free-form prose the caller must parse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .log import Logger, NULL
from .models import Router

_PROTOCOL_HEAD = """\
Return JSON of exactly this shape:
{
  "summary": "one sentence describing the change",
  "files": [{"path": "src/thing.py", "content": "<COMPLETE file contents>"}],
"""

_PROTOCOL_EDITS = """\
  "edits": [{"path": "src/thing.py", "find": "<exact text to replace>",
             "replace": "<the text that takes its place>"}],
"""

_PROTOCOL_TAIL = """\
  "delete": ["path/to/remove"],
  "commands": ["shell command to run after writing, e.g. a test command"],
  "dependencies": ["package names you added, with justification in summary"]
}
Rules:
- "content" is the ENTIRE file after your change, never a diff or a fragment.
- Never write outside the project directory and never use absolute paths.
- Prefer the standard library. Justify every new dependency in the summary.
- Omit any key you are not using.
"""

_EDIT_RULES = """\
- Use "edits" to change part of a file that already exists. "find" must be text
  copied character for character from the file as shown to you, and long enough
  to appear exactly once in it; "replace" is what it becomes. An edit that does
  not match is discarded and you will be told so.
- Use "files" for a file you are creating, or when you are genuinely rewriting a
  whole small file. Re-emitting a large file to change a few lines is slow and
  tends to lose code you did not mean to touch.
"""


def file_protocol(edits: bool = True) -> str:
    """The change format agents must answer in."""
    return (_PROTOCOL_HEAD + (_PROTOCOL_EDITS if edits else "") + _PROTOCOL_TAIL
            + (_EDIT_RULES if edits else ""))


#: The whole-file format, kept as a name for callers that only ever write files.
FILE_PROTOCOL = file_protocol(edits=False)


@dataclass
class Agent:
    """A role: a system prompt plus the task kind that routes it to a model."""

    name: str
    task: str
    system: str
    router: Router
    log: Logger = NULL

    def ask(self, prompt: str, **kw: Any) -> str:
        self.log.info("agent.ask", agent=self.name, model=self.router.describe(self.task))
        return self.router.complete(self.task, self.system, prompt, **kw)

    def ask_json(self, prompt: str, **kw: Any) -> Any:
        self.log.info("agent.ask_json", agent=self.name,
                      model=self.router.describe(self.task))
        return self.router.complete_json(self.task, self.system, prompt, **kw)


BASE = (
    "You are part of Hermes, an autonomous software factory. You work without "
    "human supervision: never ask questions, never wait for approval, never "
    "defer a decision. When information is missing, choose a sensible default "
    "and state the assumption. Be concrete and terse."
)

PROMPTS = {
    "architect": (
        f"{BASE} You are the Architect. You own system architecture, interfaces, "
        "technology selection and decomposition. You choose in this order: "
        "correctness, security, simplicity, maintainability, reproducibility, "
        "performance, implementation cost. You avoid unnecessary complexity and "
        "reject dependencies that a few lines of standard library would cover."
    ),
    "researcher": (
        f"{BASE} You are the Researcher. You use the Feynman method: explain the "
        "problem in plain language, name what is genuinely unknown, attack the "
        "unknowns, challenge your own assumptions, and convert findings into "
        "actionable decisions, hypotheses and experiments. You never merely "
        "summarise; every finding must change what gets built. Distinguish clearly "
        "between what you know, what you infer, and what must be measured."
    ),
    "implementer": (
        f"{BASE} You are the Implementer. You write complete, working, idiomatic "
        "code that compiles and passes tests on the first try. No placeholders, no "
        "TODOs, no stub functions, no 'implementation left as an exercise'. Match "
        "the existing code's style. Handle errors at trust boundaries."
    ),
    "tester": (
        f"{BASE} You are the Tester. You write tests that genuinely fail when the "
        "code is wrong: real assertions on real behaviour, edge cases, boundary "
        "values, error paths. You never write a test that passes trivially or "
        "asserts only that a function is callable."
    ),
    "reviewer": (
        f"{BASE} You are the Reviewer. You judge correctness, maintainability and "
        "security. You look for: unhandled errors, silent failures, injection and "
        "unsafe filesystem access, resource leaks, incorrect edge-case handling, "
        "and code that does not actually satisfy the stated acceptance criteria. "
        "You approve work that is correct and adequate; you do not demand polish."
    ),
    "benchmarker": (
        f"{BASE} You are the Benchmark Engineer. You build deterministic, "
        "reproducible measurements. A benchmark must print a single machine-readable "
        "score line, warm up before measuring, use fixed seeds and fixed inputs, and "
        "produce the same number twice on unchanged code."
    ),
    "evolver": (
        f"{BASE} You are the Evolution Agent. You propose a single focused change to "
        "improve a measured objective without breaking correctness. You explain the "
        "mechanism by which your change is faster or better, not merely that it is. "
        "Algorithmic and data-structure changes beat micro-optimisation."
    ),
    "documenter": (
        f"{BASE} You are the Documentation Agent. You document what the code "
        "actually does, verified against the source you are given. You never "
        "document features that do not exist. Examples must be runnable."
    ),
    "planner": (
        f"{BASE} You are the Planner. You decompose objectives into tickets small "
        "enough that one agent can implement, test and verify each in a single pass. "
        "Every ticket carries concrete acceptance criteria and a shell command that "
        "proves it."
    ),
    "triager": (
        f"{BASE} You are the Failure Triager. You diagnose failures from build and "
        "test output, classify the root cause, and prescribe the minimum repair. You "
        "distinguish a trivial slip from a wrong implementation from a wrong "
        "architecture from a genuine knowledge gap."
    ),
    "summariser": (
        f"{BASE} You are the Summariser. You compress technical material without "
        "losing a decision, a number, a name, a constraint or a rejected "
        "alternative. You keep what changes what gets built and drop the rest. "
        "You never add anything the source did not say."
    ),
}

ROUTING = {
    "architect": "architecture",
    "researcher": "research",
    "implementer": "coding",
    "tester": "coding",
    "reviewer": "review",
    "benchmarker": "coding",
    "evolver": "evolution",
    "documenter": "coding",
    "planner": "architecture",
    "triager": "review",
    "summariser": "simple",
}


def build(router: Router, log: Logger = NULL) -> dict[str, Agent]:
    """Instantiate every role against the configured router."""
    return {
        name: Agent(name=name, task=ROUTING[name], system=system, router=router, log=log)
        for name, system in PROMPTS.items()
    }


# -- shared prompt fragments -------------------------------------------------

def context_block(state, tree: str = "", extra: str = "") -> str:
    """Project orientation for any agent.

    Ordered stable-first. The identity, architecture and commands change at
    most once per project, while the file listing changes after every ticket,
    so putting the settled part first lets a local inference server reuse the
    cached KV prefix across the many calls that share it. Callers holding
    something more volatile still (a failing test log, a repair instruction)
    should append it after this block rather than pass it in.
    """
    spec = state.spec or {}
    plan = state.plan or {}
    return (
        f"PROJECT: {spec.get('name', state.name)}\n"
        f"OBJECTIVE: {spec.get('objective', state.idea)}\n"
        f"LANGUAGE/STACK: {', '.join(spec.get('technologies', []) or ['unspecified'])}\n"
        f"ARCHITECTURE: {plan.get('architecture', 'not yet defined')}\n"
        f"BUILD: {(plan.get('commands') or {}).get('build', 'n/a')}\n"
        f"TEST: {(plan.get('commands') or {}).get('test', 'n/a')}\n"
        + (f"\n{extra}\n" if extra else "")
        + (f"\nFILES:\n{tree}\n" if tree else "")
    )


def ticket_block(ticket) -> str:
    return (
        f"TICKET {ticket.id}: {ticket.title}\n"
        f"KIND: {ticket.kind}\n"
        f"OBJECTIVE: {ticket.objective or ticket.title}\n"
        f"CONTEXT: {ticket.context or '(none)'}\n"
        f"ACCEPTANCE CRITERIA:\n"
        + ("\n".join(f"  - {a}" for a in ticket.acceptance) or "  - (none stated)")
        + "\nVALIDATION COMMANDS:\n"
        + ("\n".join(f"  - {v}" for v in ticket.validation) or "  - (choose one)")
        + "\nEXPECTED ARTIFACTS:\n"
        + ("\n".join(f"  - {a}" for a in ticket.artifacts) or "  - (your judgement)")
    )
