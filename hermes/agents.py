"""Specialised agent roles.

Agents are stateless: everything they need is passed in from project state,
so any agent can be re-run after a crash without replaying a conversation.
Each returns plain Python data, never free-form prose the caller must parse.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .contextdb import TRUSTED, UNTRUSTED
from .log import Logger, NULL
from .models import Allowance, PromptTooLarge, Router
from .schema import STRING, arr, obj

_PROTOCOL_HEAD = """\
Return JSON of exactly this shape:
{
  "summary": "one sentence describing the change",
  "files": [{"path": "src/thing.py", "content": "<COMPLETE file contents>"}],
"""

_PROTOCOL_EDITS = """\
  "edits": [{"path": "src/thing.py", "find": "<exact text to replace>",
             "replace": "<the text that takes its place>",
             "symbol": "<optional: the function or class it is inside>"}],
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

#: How to phrase a change as a patch. Shared with the evolver, which edits the
#: project's largest and hottest files and so has the most to lose by rewriting
#: them whole.
EDIT_RULES = """\
- Use "edits" to change part of a file that already exists. "find" must be text
  copied character for character from the file as shown to you, and long enough
  to appear exactly once in it; "replace" is what it becomes. An edit that does
  not match is discarded and you will be told so.
- Name the enclosing function or class in "symbol" when the file has several
  places the same text could be. That is what decides which one you meant.
- Use "files" for a file you are creating, or when you are genuinely rewriting a
  whole small file. Re-emitting a large file to change a few lines is slow and
  tends to lose code you did not mean to touch.
"""


def file_protocol(edits: bool = True) -> str:
    """The change format agents must answer in."""
    return (_PROTOCOL_HEAD + (_PROTOCOL_EDITS if edits else "") + _PROTOCOL_TAIL
            + (EDIT_RULES if edits else ""))


#: The whole-file format, kept as a name for callers that only ever write files.
FILE_PROTOCOL = file_protocol(edits=False)

_FILE = obj({"path": STRING, "content": STRING}, required=("path", "content"))
_EDIT = obj({"path": STRING, "find": STRING, "replace": STRING, "symbol": STRING},
            required=("path", "find", "replace"))


def change_schema(edits: bool = True) -> dict:
    """The shape of a change, matching :func:`file_protocol`.

    ``path`` and ``content`` are required *within* a file rather than at the
    top: the change itself may legitimately be nothing but a command or a
    delete, but a file entry missing its body is the one failure that used to
    cost a whole generation -- the model writes ``{"path": "src/x.py"}`` and
    means "unchanged", and the applier cannot tell that from a truncation.

    ``summary`` is required because it is what the commit message, the journal
    and the next ticket's context are all written from.
    """
    properties = {
        "summary": STRING,
        "files": arr(_FILE),
        "delete": arr(),
        "commands": arr(),
        "dependencies": arr(),
    }
    if edits:
        properties["edits"] = arr(_EDIT)
    return obj(properties, required=("summary",))


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

    def allowance(self) -> Allowance:
        """What this role's model gives it to spend on one prompt."""
        return self.router.allowance(self.task)

    def prompt(self, log: Logger | None = None) -> "Prompt":
        """An empty prompt already sized to this agent's budget."""
        return Prompt(self, log=log)


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


def escalate(agents: dict[str, Agent], base: Agent, role: str,
             log: Logger = NULL) -> Agent:
    """The role a harder attempt should ask, on the model already resident.

    Escalation names a role, not a model. Routing a retry to a second model
    means unloading the first one and loading the second -- the reload
    :meth:`Router.preflight` warns about -- which on a single-GPU host costs
    more than the retry it is meant to rescue. So when the stronger role
    routes elsewhere, its system prompt is asked of the model the ticket is
    already using: what changes is how the work is framed, not the weights.

    An unknown role leaves the base agent alone; a ladder naming a role the
    factory does not have should cost a log line, not the ticket.
    """
    agent = agents.get(role)
    if agent is None:
        log.warn("agent.escalation_unknown", role=role)
        return base
    if agent is base or agent.router.describe(agent.task) == base.router.describe(base.task):
        return agent
    log.info("agent.escalation_pinned", role=role, task=base.task,
             model=base.router.describe(base.task))
    return replace(agent, task=base.task)


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


# -- prompt assembly ---------------------------------------------------------

@dataclass
class Block:
    """One labelled piece of a prompt, and what to do when it will not fit.

    A block that a caller must size itself -- file contents, a failure log --
    is added empty, asked for its :meth:`room`, and filled.
    """

    name: str
    text: str = ""
    priority: int = 50
    #: Characters below which a trimmed copy is not worth its room. Without
    #: one, the block is kept whole or dropped whole.
    floor: int = 0
    #: Keep the end rather than the start. True for anything where the last
    #: lines carry the meaning, such as a traceback.
    tail: bool = False
    #: The prompt cannot be built without it: never trimmed, never dropped.
    required: bool = False
    #: Where this text came from. A block holding anything fetched off the
    #: network is marked :data:`hermes.contextdb.UNTRUSTED`, and that marking
    #: is what :class:`hermes.tools.Lookup` refuses to give tools to.
    trust: str = TRUSTED
    owner: "Prompt | None" = None

    def room(self) -> int:
        """Characters free for this block once everything above it is in."""
        return self.owner.room(self.priority) if self.owner else 0

    def fill(self, text: str) -> "Block":
        self.text = text
        return self


class Prompt:
    """An agent prompt assembled inside its model's token budget.

    Blocks are added in the order they should be *read* -- stable first, so a
    local server can reuse the cached prefix across the calls that share it --
    and carry a separate priority saying what survives when they do not all
    fit. Spending in priority order and dropping whole blocks is the point:
    trimming every block a little leaves half a file listing, half the
    journal and half the source the ticket is actually about, and the last of
    those is the one that makes the prompt unanswerable.
    """

    MARKER = "… (trimmed to fit the model context)"

    #: Characters the blank line between blocks costs.
    SEPARATOR = 2

    def __init__(self, agent: Agent, log: Logger | None = None):
        self.allowance = agent.allowance()
        self.log = agent.log if log is None else log
        self.blocks: list[Block] = []
        # The role's system prompt is charged against the same context, and
        # complete_json adds a line to it, so the user turn cannot have the
        # whole allowance.
        self.limit = max(0, self.allowance.prompt_chars - len(agent.system) - 200)

    def add(self, name: str, text: str = "", *, priority: int = 50, floor: int = 0,
            tail: bool = False, required: bool = False,
            trust: str = TRUSTED) -> Block:
        block = Block(name, text, priority, floor, tail, required, trust, owner=self)
        self.blocks.append(block)
        return block

    @property
    def untrusted(self) -> tuple[str, ...]:
        """The blocks carrying text the factory did not write.

        A prompt that has any is a prompt an attacker has a sentence in, and
        :class:`hermes.tools.Lookup` will not attach a toolbox to it. The
        property is on the prompt rather than on the caller because the
        caller is exactly who cannot be relied on to remember.
        """
        return tuple(b.name for b in self.blocks if b.trust != TRUSTED)

    def room(self, priority: int) -> int:
        """Characters still free for a block of *priority*.

        Counts everything that outranks it, whether or not it has been added
        yet -- so a caller that sizes its own content gets the same answer
        wherever in the assembly it asks.
        """
        spent = sum(len(b.text) + self.SEPARATOR for b in self.blocks
                    if (b.required or b.priority < priority) and b.text)
        return max(0, self.limit - spent - self.SEPARATOR)

    def render(self) -> str:
        """Spend the budget, then join what survived in reading order."""
        limit = self.limit
        required = [b for b in self.blocks if b.required]
        room = limit - sum(len(b.text) + self.SEPARATOR for b in required if b.text)
        if room < 0:
            raise PromptTooLarge(
                f"task {self.allowance.task!r}: the blocks that cannot be dropped "
                f"are {-room} characters over a budget of {limit} "
                f"({self.allowance.prompt_tokens} tokens of a "
                f"{self.allowance.context}-token context): "
                + ", ".join(f"{b.name} {len(b.text)}c" for b in required)
            )
        kept = {id(b): b.text for b in required}
        for block in sorted((b for b in self.blocks if not b.required),
                            key=lambda b: b.priority):
            if not block.text:
                continue
            cost = len(block.text) + self.SEPARATOR
            if cost <= room:
                kept[id(block)] = block.text
                room -= cost
            elif block.floor and room - self.SEPARATOR >= block.floor:
                kept[id(block)] = self._trim(block, room - self.SEPARATOR)
                self.log.warn("prompt.block_trimmed", task=self.allowance.task,
                              block=block.name, had=len(block.text),
                              kept=len(kept[id(block)]))
                room -= len(kept[id(block)]) + self.SEPARATOR
            else:
                self.log.warn("prompt.block_dropped", task=self.allowance.task,
                              block=block.name, chars=len(block.text), room=room)
        return "\n\n".join(text for text in
                            (kept.get(id(b), "") for b in self.blocks) if text)

    def _trim(self, block: Block, room: int) -> str:
        body = room - len(self.MARKER) - 1
        if body <= 0:
            return ""
        if block.tail:
            return f"{self.MARKER}\n{block.text[-body:]}"
        return f"{block.text[:body]}\n{self.MARKER}"
