"""Running the experiments the research loop proposes.

The Feynman loop in :mod:`hermes.research` stops at a hypothesis: a testable
claim plus a sentence about how it could be tested. This module is the other
half of section 7 -- build the minimal experiment, measure, update the
understanding. It asks for the smallest program that settles the claim, runs
it in the sandbox, and writes the record section 20 asks for, complete enough
that the measurement can be repeated by someone who never runs Hermes.

Two things here are deliberate. A verdict is about the experiment, not about
the hypothesis: an experiment that ran and refuted its claim succeeded, and is
worth more to the project than one that was never run, so a refutation
completes its ticket. And nothing is ever written over: every run gets its own
identifier and its own directory, because an experiment whose earlier result
was overwritten by a later one cannot be checked against it.
"""
from __future__ import annotations

import json
import re
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agents import Prompt, context_block
from .implement import Change
from .log import Logger, NULL, scope
from .research import ResearchLog
from .sandbox import Sandbox
from .tools import Lookup, Toolbox
from .verify import Gate, Verification, scan_unsafe
from .workspace import slugify

#: How a program reports what it measured. One line, machine-readable, so the
#: number is the program's claim rather than something parsed out of prose.
RESULT_LINE = re.compile(r"^\s*(?:HERMES_)?RESULT\s*[:=]\s*(.+?)\s*$", re.MULTILINE)
_PAIR = re.compile(r"([A-Za-z_][\w .-]*?)\s*[=:]\s*"
                   r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

PROTOCOL = (
    "Report the measurement on its own line, in this exact form:\n"
    '  RESULT: {"name": <number>, "other_name": <number>}\n'
    "one entry per quantity you measured, printed last, with nothing else on "
    "that line. Everything else the program prints is kept as its log."
)


def measurements(output: str) -> dict:
    """Every ``RESULT:`` line a program printed, merged, later lines winning."""
    found: dict = {}
    for raw in RESULT_LINE.findall(output or ""):
        found.update(_parse(raw))
    return found


def _parse(raw: str) -> dict:
    """One RESULT payload, however close to the protocol the program came.

    The protocol asks for a JSON object, and a program that obeyed it is read
    exactly. The fallbacks exist because a program that measured the right
    thing and printed it as ``ns_per_op=4.2`` has still measured the right
    thing, and discarding that would throw away the experiment over its
    formatting.
    """
    try:
        value = json.loads(raw)
    except ValueError:
        value = None
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"value": value}
    pairs = {name.strip(): float(number) for name, number in _PAIR.findall(raw)}
    if pairs:
        return pairs
    numbers = _NUMBER.findall(raw)
    return {"value": float(numbers[-1])} if numbers else {}


def aggregate(runs: list["Run"]) -> dict:
    """The one set of numbers that represents every run of this experiment.

    The median for a quantity every run reported as a number, for the reason
    :mod:`hermes.evolve` uses it: a shared machine can only ever make a run
    slower, so a mean reports the interference rather than the code. Anything
    non-numeric is taken from the last run, which is the only reading of it
    that is not an invention.
    """
    merged: dict = {}
    for run in runs:
        for key, value in run.measurements.items():
            merged.setdefault(key, []).append(value)
    result = {}
    for key, values in merged.items():
        numeric = [v for v in values
                   if isinstance(v, (int, float)) and not isinstance(v, bool)]
        result[key] = (statistics.median(numeric) if len(numeric) == len(values)
                       and numeric else values[-1])
    return result


@dataclass
class Run:
    """One execution of the experiment's program."""

    at: float = field(default_factory=time.time)
    seconds: float = 0.0
    exit_code: int = 0
    timed_out: bool = False
    measurements: dict = field(default_factory=dict)
    log: str = ""               # path to the full output, relative to the project

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and bool(self.measurements)


@dataclass
class Record:
    """Everything section 20 requires of an experiment, in one document."""

    id: str
    hypothesis: str
    test: str = ""
    directory: str = ""
    design: str = ""
    command: str = ""
    parameters: dict = field(default_factory=dict)
    configuration: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    code_version: str = ""
    models: dict = field(default_factory=dict)
    ticket: str = ""
    at: float = field(default_factory=time.time)
    runs: list[Run] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)
    verdict: str = "inconclusive"     # supported | refuted | inconclusive
    finding: str = ""
    confidence: str = "uncertain"
    implications: list[str] = field(default_factory=list)
    next_step: str = ""
    artifacts: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def ran(self) -> bool:
        """True when a measurement exists, whatever it says about the claim."""
        return bool(self.measurements)

    def render(self) -> str:
        def block(title: str, body: str) -> str:
            return f"## {title}\n{body}\n\n" if body else ""

        runs = "\n".join(
            f"- run {n}: exit {r.exit_code}"
            + (" (timed out)" if r.timed_out else "")
            + f", {r.seconds:.1f}s, {json.dumps(r.measurements)}"
            + (f" — log: `{r.log}`" if r.log else "")
            for n, r in enumerate(self.runs, 1)) or "- (never ran)"
        return (
            f"# {self.id}: {self.hypothesis}\n\n"
            f"_{time.strftime('%Y-%m-%d %H:%M', time.localtime(self.at))}_"
            + (f" · ticket {self.ticket}" if self.ticket else "")
            + (f" · code {self.code_version}" if self.code_version else "")
            + "\n\n"
            + f"**Verdict:** {self.verdict} ({self.confidence})\n\n"
            + block("Finding", self.finding)
            + block("Hypothesis", f"{self.hypothesis}\n\n_Proposed test:_ "
                                  f"{self.test or 'none given'}")
            + block("Design", self.design)
            + block("Measurement", f"```json\n{json.dumps(self.measurements, indent=2)}"
                                   f"\n```")
            + block("Runs", runs)
            + block("Implications", "\n".join(f"- {i}" for i in self.implications))
            + block("Next", self.next_step)
            + block("Reproducing", f"```console\n$ cd {self.directory}\n"
                                   f"$ {self.command or '(no command)'}\n```\n"
                                   f"Parameters: `{json.dumps(self.parameters)}`\n\n"
                                   f"Environment: `{self.environment.get('probe', '')}`")
            + block("Refused", "\n".join(f"- {r}" for r in self.refused))
            + block("Notes", self.notes)
        )


class Lab:
    """Builds, runs and records the experiments that settle hypotheses."""

    def __init__(self, sandbox: Sandbox, agents, state, cfg: dict | None = None,
                 log: Logger = NULL, governor=None):
        self.sandbox = sandbox
        self.agents = agents
        self.state = state
        self.cfg = cfg or {}
        self.log = log
        #: Optional :class:`~hermes.resources.Governor`. An experiment that
        #: measures time measures the machine, so it wants the machine still.
        self.governor = governor
        self.root = Path(sandbox.root)

    # -- allocation ------------------------------------------------------
    def allocate(self, hypothesis: str) -> tuple[str, str]:
        """A fresh identifier and directory for one experiment.

        The identifier is taken from what is already on disk rather than from
        state, so running the same hypothesis again lands beside the earlier
        result instead of on top of it, even across a resumed project.
        """
        directory = self.root / "experiments"
        directory.mkdir(parents=True, exist_ok=True)
        used = [int(match.group(1)) for path in directory.iterdir()
                if path.is_dir() and (match := re.match(r"E(\d+)-", path.name))]
        ident = f"E{max(used, default=0) + 1:03d}"
        return ident, f"experiments/{ident}-{slugify(hypothesis)}"

    # -- the experiment --------------------------------------------------
    def conduct(self, hypothesis: str, *, test: str = "", brief: str = "",
                ticket: str = "", tree: str = "", toolbox: Toolbox | None = None,
                version: str = "") -> Record:
        """Design, run and record one experiment. Never raises on a bad program."""
        ident, directory = self.allocate(hypothesis)
        record = Record(id=ident, hypothesis=hypothesis[:1000], test=test[:1000],
                        directory=directory, ticket=ticket, code_version=version,
                        configuration=self._configuration(),
                        models={"design": self._model("benchmarker"),
                                "interpretation": self._model("researcher")})
        attempts = max(1, int(self.cfg.get("experiment_attempts", 2)))
        failure, written = "", []
        for attempt in range(1, attempts + 1):
            with scope(experiment=ident, attempt=attempt):
                self.log.info("experiment.design", hypothesis=hypothesis[:80])
                _discard(self.sandbox, written)
                payload = self._design(hypothesis, test, brief, tree, toolbox, failure)
                written, refused = self._write(directory, payload)
                record.refused = refused
                failure = self._unusable(payload, written)
                if failure:
                    self.log.warn("experiment.unusable", reason=failure[:120])
                    continue
                record.design = str(payload.get("design", ""))[:2000]
                record.parameters = payload.get("parameters") or {}
                record.command = str(payload.get("command", "")).strip()
                record.artifacts = written
                record.environment = self._environment(directory)
                record.runs = self._measure(directory, record.command)
                record.measurements = aggregate(record.runs)
                if record.ran:
                    break
                failure = self._why(record.runs)
                self.log.warn("experiment.no_measurement", detail=failure[:200])

        if record.ran:
            self._interpret(record)
        else:
            record.notes = failure or "the experiment never produced a measurement"
        self._publish(record)
        self.log.info("experiment.done", id=record.id, verdict=record.verdict,
                      measured=len(record.measurements), runs=len(record.runs))
        return record

    # -- steps -----------------------------------------------------------
    def _design(self, hypothesis: str, test: str, brief: str, tree: str,
                toolbox: Toolbox | None, failure: str) -> dict:
        """Ask for the smallest program that settles the claim by measuring it.

        The Benchmark Engineer rather than the Implementer: what is wanted is
        a deterministic measurement with a machine-readable result, which is
        that role's whole job. It may read the workspace, because a hypothesis
        about this project's code cannot be tested without seeing it.
        """
        builder = self.agents["benchmarker"]
        timeout = int(self.cfg.get("experiment_timeout", 300))
        prompt = Prompt(builder, log=self.log)
        prompt.add("context", context_block(self.state), required=True)
        prompt.add("hypothesis",
                   f"HYPOTHESIS TO SETTLE: {hypothesis}\n"
                   f"TEST PROPOSED BY RESEARCH: {test or '(none; design one)'}",
                   required=True)
        prompt.add("brief", brief, priority=30, floor=400)
        prompt.add("tree", f"FILES:\n{tree}", priority=50, floor=400)
        if failure:
            prompt.add("failure",
                       f"YOUR LAST PROGRAM DID NOT PRODUCE A MEASUREMENT:\n{failure}\n"
                       "Fix the cause; do not resend the same program.",
                       priority=10, floor=800, tail=True)
        prompt.add("instructions",
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
            + PROTOCOL + "\n\nReturn JSON:\n"
            "{\n"
            '  "design": "what is measured, and why that settles the hypothesis",\n'
            '  "parameters": {"name": value},\n'
            '  "files": [{"path": "run.py", "content": "<COMPLETE file contents>"}],\n'
            '  "command": "python3 run.py",\n'
            '  "supports": "the result that would support the hypothesis",\n'
            '  "refutes": "the result that would refute it"\n'
            "}", required=True)
        return Lookup(builder, prompt, toolbox, log=self.log).ask()

    def _write(self, directory: str, payload: dict) -> tuple[list[str], list[str]]:
        """Write the program inside its own directory, and nowhere else.

        An experiment must not be able to edit the project it is measuring: a
        path that leaves the experiment directory is refused rather than
        clamped, because a program that meant to write elsewhere is not the
        program that was asked for.
        """
        written, refused = [], []
        for item in payload.get("files") or []:
            if not isinstance(item, dict):
                continue
            relative = _inside(item.get("path", ""))
            content = item.get("content")
            if not relative or content is None:
                refused.append(f"{item.get('path')}: outside the experiment directory")
                continue
            path = f"{directory}/{relative}"
            self.sandbox.write(path, content if isinstance(content, str)
                               else json.dumps(content, indent=2))
            written.append(path)
        for entry in refused:
            self.log.warn("experiment.rejected", detail=entry[:160])
        return written, refused

    def _unusable(self, payload: dict, written: list[str]) -> str:
        """Why this program cannot be run, or empty if it can be."""
        if not written:
            return ("you returned no files inside the experiment directory; "
                    "return the complete program")
        if not str(payload.get("command", "")).strip():
            return "you returned no command to run the program with"
        hits = scan_unsafe(self.root, written)
        if hits:
            # The program is model-written and about to be executed. Refusing
            # it here costs one attempt; running it costs whatever it does.
            _discard(self.sandbox, written)
            return ("the program was refused by the security scan: "
                    + "; ".join(hits[:3]))
        return ""

    def _measure(self, directory: str, command: str) -> list[Run]:
        """Run the program, keeping each run's output as its own file."""
        repeats = max(1, int(self.cfg.get("experiment_repeats", 1)))
        timeout = int(self.cfg.get("experiment_timeout", 300))
        runs: list[Run] = []
        quiet = self.governor.measuring() if self.governor else nullcontext()
        with quiet:
            for index in range(1, repeats + 1):
                result = self.sandbox.run(command, timeout=timeout, cwd=directory)
                output = result.stdout + ("\n" + result.stderr if result.stderr else "")
                path = f"{directory}/runs/run-{index:03d}.log"
                self.sandbox.write(path, f"$ {command}\n[exit {result.exit_code}, "
                                         f"{result.seconds:.1f}s]\n{output}")
                runs.append(Run(seconds=round(result.seconds, 3),
                                exit_code=result.exit_code,
                                timed_out=result.timed_out,
                                measurements=measurements(output), log=path))
                self.log.info("experiment.run", n=index, exit=result.exit_code,
                              seconds=round(result.seconds, 1),
                              measured=len(runs[-1].measurements))
        return runs

    def _interpret(self, record: Record) -> None:
        """Ask what the numbers say about the claim, shown the program too.

        The program is in the prompt because the question that decides the
        verdict is whether what ran measures what was claimed; a number alone
        cannot answer that, and a researcher shown only the number will
        believe it.
        """
        researcher = self.agents["researcher"]
        prompt = Prompt(researcher, log=self.log)
        prompt.add("context", context_block(self.state), priority=40, floor=400)
        prompt.add("hypothesis",
                   f"HYPOTHESIS: {record.hypothesis}\n"
                   f"DESIGN: {record.design or '(none recorded)'}", required=True)
        prompt.add("measurement",
                   "MEASURED:\n" + json.dumps(record.measurements, indent=2)
                   + (f"\n\nACROSS {len(record.runs)} RUNS:\n"
                      + "\n".join(json.dumps(r.measurements) for r in record.runs)
                      if len(record.runs) > 1 else ""), required=True)
        program = prompt.add("program", priority=20, floor=800)
        prompt.add("output", "OUTPUT OF THE LAST RUN:\n" + self._tail(record),
                   priority=30, floor=400)
        prompt.add("instructions",
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
            "}", required=True)
        header = "THE PROGRAM THAT PRODUCED IT:\n"
        program.fill(header + self._program(record, max(0, program.room() - len(header))))
        verdict = researcher.ask_json(prompt.render())
        stated = str(verdict.get("verdict", "")).lower().strip()
        record.verdict = (stated if stated in ("supported", "refuted", "inconclusive")
                          else "inconclusive")
        record.finding = str(verdict.get("finding", ""))[:2000]
        record.confidence = str(verdict.get("confidence", "uncertain"))[:40]
        record.implications = [str(i)[:300] for i in (verdict.get("implications") or [])][:6]
        record.next_step = str(verdict.get("next", ""))[:500]

    # -- recording -------------------------------------------------------
    def _publish(self, record: Record) -> None:
        """Write the record, the metric and the research feedback."""
        directory = self.root / record.directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "record.json").write_text(
            json.dumps(asdict(record), indent=2, default=str))
        (directory / "README.md").write_text(record.render())
        record.artifacts = sorted(set(record.artifacts + [
            f"{record.directory}/record.json", f"{record.directory}/README.md"]))

        metrics = self.root / ".hermes" / "metrics" / "experiments.jsonl"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        with metrics.open("a") as handle:
            handle.write(json.dumps({
                "id": record.id, "at": record.at, "ticket": record.ticket,
                "hypothesis": record.hypothesis, "verdict": record.verdict,
                "confidence": record.confidence, "measurements": record.measurements,
                "runs": len(record.runs), "code_version": record.code_version,
                "record": f"{record.directory}/record.json",
            }, default=str) + "\n")
        self._settle(record)

    def _settle(self, record: Record) -> None:
        """Put the measurement back where the next research round will read it.

        This is the edge section 7 draws and the code did not have. A round
        loads what earlier rounds established before it chooses its unknowns,
        so an experiment written here is what stops the next round asking the
        question this one just answered.
        """
        finding = record.finding or (
            f"the experiment did not produce a measurement: {record.notes}")
        settled = record.ran and record.verdict != "inconclusive"
        ResearchLog(self.root).add(f"experiment {record.id}: {record.hypothesis}"[:300], {
            "experiment": record.id,
            "record": f"{record.directory}/record.json",
            "measurements": record.measurements,
            "findings": [{
                "question": record.hypothesis,
                "answer": f"{record.verdict}: {finding}",
                "confidence": record.confidence if record.ran else "uncertain",
                "sources": [],
                "implications": record.implications,
            }],
            "conclusions": ([f"{record.hypothesis} — {record.verdict} by "
                             f"experiment {record.id}: {finding}"] if settled else []),
            "decisions": [], "hypotheses": [], "tickets": [],
            "remaining_unknowns": [] if settled else [record.next_step or record.hypothesis],
        })

    # -- context ---------------------------------------------------------
    def _configuration(self) -> dict:
        """What was set when this ran, so a repeat can be set up the same way."""
        sandbox = self.sandbox.cfg
        return {
            "runtime": self.sandbox.runtime,
            "image": sandbox.get("image"),
            "cpus": sandbox.get("cpus"),
            "memory": sandbox.get("memory"),
            "network": sandbox.get("network"),
            "repeats": max(1, int(self.cfg.get("experiment_repeats", 1))),
            "timeout": int(self.cfg.get("experiment_timeout", 300)),
        }

    def _environment(self, directory: str) -> dict:
        """The box the measurement was taken on, as the box itself reports it."""
        probe = self.sandbox.run(
            "uname -sr 2>/dev/null; python3 --version 2>&1; "
            "echo cpus=$(nproc 2>/dev/null || echo unknown)",
            timeout=60, cwd=directory)
        return {"probe": " | ".join(probe.stdout.split()) or "unavailable",
                "workdir": self.sandbox.workdir, "at": time.time()}

    def _model(self, role: str) -> str:
        agent = self.agents.get(role)
        try:
            return agent.router.describe(agent.task)
        except Exception:  # noqa: BLE001 - a double without a router still runs
            return getattr(agent, "task", role)

    def _program(self, record: Record, budget: int) -> str:
        chunks, used = [], 0
        for path in record.artifacts:
            if path.endswith((".json", ".md", ".log")):
                continue
            text = self.sandbox.read(path)[: max(0, budget - used)]
            if not text:
                break
            used += len(text)
            chunks.append(f"--- {path} ---\n{text}")
        return "\n\n".join(chunks) or "(the program could not be read back)"

    def _tail(self, record: Record, limit: int = 2000) -> str:
        if not record.runs:
            return "(the program never ran)"
        return self.sandbox.read(record.runs[-1].log)[-limit:] or "(no output)"

    @staticmethod
    def _why(runs: list[Run]) -> str:
        """Why a run produced no measurement, phrased for the next attempt."""
        if not runs:
            return "the program was never run"
        last = runs[-1]
        if last.timed_out:
            return "the program timed out; make the experiment smaller"
        if last.exit_code:
            return f"the program exited {last.exit_code}"
        return ("the program ran but printed no RESULT line, so nothing was "
                "measured. " + PROTOCOL)


def _inside(path) -> str:
    """The path relative to the experiment directory, or empty if it escapes.

    An absolute path is refused rather than reinterpreted as a relative one.
    Stripping its leading slash would write a file the program then cannot
    find, and call that success; the program asked for somewhere it may not
    write, and the honest answer is to say so.
    """
    clean = str(path or "").strip().replace("\\", "/")
    if not clean or clean.startswith("/") or Path(clean).is_absolute():
        return ""
    if clean.startswith("../") or "/../" in clean or clean.endswith("/.."):
        return ""
    return clean


def _discard(sandbox: Sandbox, paths: list[str]) -> None:
    """Remove a rejected attempt's files so the next one starts clean."""
    for path in paths:
        try:
            sandbox.resolve(path).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 - nothing here is worth failing over
            continue


def run_ticket(lab: Lab, ticket, *, tree: str = "", toolbox: Toolbox | None = None,
               version: str = "") -> dict:
    """Run an experiment ticket, in the verdict shape ``execute_ticket`` returns.

    The ticket is completed when the experiment produced a measurement,
    whatever the measurement says: a hypothesis that was refuted has been
    settled, and re-running it would only refute it again. It fails when there
    is no measurement at all, which is the one case a further attempt can fix.
    """
    brief = (f"TICKET {ticket.id}: {ticket.title}\n"
             f"CONTEXT: {ticket.context or '(none)'}\n"
             "ACCEPTANCE CRITERIA:\n"
             + ("\n".join(f"  - {a}" for a in ticket.acceptance) or "  - (none stated)"))
    record = lab.conduct(ticket.objective or ticket.title,
                         test=_proposed_test(ticket), brief=brief, ticket=ticket.id,
                         tree=tree, toolbox=toolbox, version=version)
    verification = Verification(gates=[Gate(
        f"experiment:{record.id}", record.ran,
        f"{record.verdict}: {record.finding[:300]}" if record.ran
        else record.notes[:300])])
    change = Change(summary=f"{record.id} {record.verdict}: "
                            f"{(record.finding or record.notes)[:200]}",
                    written=list(record.artifacts))
    if record.ran:
        return {"status": "completed", "verification": verification,
                "change": change, "record": record,
                "review": {"approved": True, "advisory": record.implications[:2]}}
    return {"status": "failed", "action": "repair", "record": record,
            "verification": verification, "change": change,
            "diagnosis": {"diagnosis": record.notes or "no measurement",
                          "classification": "implementation", "action": "repair"}}


def _proposed_test(ticket) -> str:
    """The test research suggested, wherever the ticket carried it."""
    for line in (ticket.context or "").splitlines():
        if line.lower().startswith("proposed test:"):
            return line.split(":", 1)[1].strip()
    return "\n".join(ticket.validation or []) or "\n".join(ticket.acceptance or [])
