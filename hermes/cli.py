"""``hermes`` command line interface."""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from . import phases, workspace
from .config import Config, ConfigError, SAMPLE_CONFIG
from .journal import Journal
from .log import for_workspace, scope
from .sandbox import Sandbox, probe_resources
from .state import PHASES, State, checkpoint, restore
from .tickets import Board


def _project_root(args) -> Path:
    root = Path(args.project or ".").resolve()
    if (root / ".hermes" / "state" / "state.json").is_file():
        return root
    for candidate in sorted((Path.cwd() / "projects").glob("*")):
        if (candidate / ".hermes" / "state" / "state.json").is_file():
            return candidate.resolve()
    raise SystemExit(f"no Hermes project at {root}. Run `hermes new \"idea\"` first.")


def _config(args) -> Config:
    return Config.load(getattr(args, "config", None))


def _server_for(factory):
    """Keep the local model server up for a phase run outside `Factory.run`.

    `evolve` and `release` drive a single phase directly, and a phase that
    generates needs the server as much as a whole run does.
    """
    from .server import running
    return running(factory.config, factory.router, factory.log)


# -- commands ----------------------------------------------------------------

def cmd_new(args) -> int:
    from .factory import create
    config = _config(args)
    root = Path(args.into).resolve() if args.into else (
        Path.cwd() / "projects" / workspace.slugify(args.idea))
    if (root / ".hermes").exists() and not args.force:
        print(f"{root} already holds a project; pass --force to reuse it.", file=sys.stderr)
        return 1
    root.mkdir(parents=True, exist_ok=True)
    if args.github:
        config.data["github"]["enabled"] = True
    factory = create(root, args.idea, config, quiet=args.quiet)
    print(f"project: {root}")
    if args.no_run:
        print("run it with: hermes run --project " + str(root))
        return 0
    state = factory.run(max_iterations=args.max_iterations)
    print(f"\nfinal phase: {state.phase}")
    return 0 if state.phase == "COMPLETED" else 2


def cmd_run(args) -> int:
    from .factory import Factory
    root = _project_root(args)
    factory = Factory(root, _config(args), quiet=args.quiet)
    if factory.state.paused:
        factory.state.paused = False
    state = factory.run(max_iterations=args.max_iterations)
    print(f"final phase: {state.phase}")
    return 0 if state.phase == "COMPLETED" else 2


def cmd_status(args) -> int:
    root = _project_root(args)
    state = State.load(root)
    board = Board(root)
    if args.json:
        print(json.dumps({
            "root": str(root), "name": state.name, "phase": state.phase,
            "iteration": state.iteration, "paused": state.paused,
            "progress": board.progress(), "tickets": board.counts(),
            "metrics": state.metrics, "failures": state.failures,
        }, indent=2, default=str))
        return 0
    active = board.by_status("active")
    print(f"project   {state.name}  ({root})")
    print(f"idea      {state.idea[:100]}")
    print(f"phase     {state.phase}{'  [paused]' if state.paused else ''}")
    print(f"iteration {state.iteration}")
    print(f"progress  {board.progress():.0%}")
    if active:
        print(f"active    {active[0].id} {active[0].title[:60]}")
    if state.metrics.get("optimization"):
        metric = state.metrics["optimization"]
        print(f"objective {metric['metric']}: {metric['baseline']} -> {metric['best']}")
    if state.failures:
        print("failures  " + ", ".join(f"{k}={v}" for k, v in state.failures.items()))
    print()
    print(board.summary())
    return 0


def cmd_pause(args) -> int:
    root = _project_root(args)
    state = State.load(root)
    state.paused = True
    state.save(root)
    print("paused; resume with `hermes resume`")
    return 0


def cmd_resume(args) -> int:
    root = _project_root(args)
    state = State.load(root)
    state.paused = False
    state.save(root)
    return cmd_run(args)


def cmd_logs(args) -> int:
    root = _project_root(args)
    files = sorted((root / ".hermes" / "logs").glob("*.jsonl"))
    if not files:
        print("no logs yet")
        return 0
    path = files[-1]
    if args.follow:
        with path.open() as fh:
            fh.seek(0, 2)
            while True:
                line = fh.readline()
                if line:
                    print(_format_log(line, args.json), flush=True)
                else:
                    time.sleep(0.4)
    lines = path.read_text().splitlines()[-args.lines:]
    for line in lines:
        print(_format_log(line, args.json))
    return 0


def _format_log(line: str, raw: bool) -> str:
    if raw:
        return line.rstrip()
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return line.rstrip()
    detail = " ".join(f"{k}={v}" for k, v in record.items()
                      if k not in ("ts", "iso", "level", "event"))
    return f"{record.get('iso', '')} {record.get('level', ''):<5} {record.get('event', '')} {detail}"


def cmd_inspect(args) -> int:
    root = _project_root(args)
    state = State.load(root)
    board = Board(root)
    if args.ticket:
        ticket = board.get(args.ticket)
        if not ticket:
            print(f"no ticket {args.ticket}", file=sys.stderr)
            return 1
        print(ticket.render())
        return 0
    print(f"# {state.name}\n")
    print(f"idea: {state.idea}\nphase: {state.phase}\n")
    print("## Spec\n" + json.dumps(state.spec, indent=2)[:4000])
    print("\n## Plan\n" + json.dumps(state.plan, indent=2)[:4000])
    print("\n## Board\n" + board.summary(60))
    if state.notes:
        print("\n## Notes\n" + "\n".join(f"- {n}" for n in state.notes[-15:]))
    print("\n## Phase history")
    for entry in state.history[-15:]:
        print(f"  {entry['from']} -> {entry['to']}: {entry.get('reason', '')}")
    experiments = _experiments(root)
    if experiments:
        print("\n## Experiments\n" + "\n".join(experiments))
    recent = Journal(root).entries[-8:]
    if recent:
        print("\n## Recent work\n" + "\n".join(e.render() for e in recent))
    return 0


def _experiments(root: Path, limit: int = 10) -> list[str]:
    """What the hypotheses this project raised were settled by, one line each."""
    path = root / ".hermes" / "metrics" / "experiments.jsonl"
    if not path.is_file():
        return []
    lines = []
    for raw in path.read_text().splitlines()[-limit:]:
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        lines.append(f"  {record.get('id', '?'):<6} {record.get('verdict', '?'):<13}"
                     f" {str(record.get('hypothesis', ''))[:54]}\n"
                     f"         {json.dumps(record.get('measurements', {}))[:70]}")
    return lines


def cmd_journal(args) -> int:
    """What each agent did: the same record the agents themselves read."""
    root = _project_root(args)
    entries = Journal(root).entries
    if args.ticket:
        entries = [e for e in entries if e.ticket == args.ticket]
    entries = entries[-args.lines:]
    if not entries:
        print("no work recorded yet")
        return 0
    for entry in entries:
        print(json.dumps(asdict(entry), default=str) if args.json
              else entry.render() + "\n")
    return 0


def cmd_test(args) -> int:
    from .verify import verify_project
    root = _project_root(args)
    state = State.load(root)
    config = _config(args)
    log = for_workspace(root, config.get("logging.level"), args.quiet)
    with Sandbox(root, config.get("sandbox", {}), log) as sandbox:
        verification = verify_project(sandbox, state.plan, log=log)
    print(verification.report())
    if not verification.passed:
        print("\n" + verification.failure_output())
    return 0 if verification.passed else 1


def cmd_benchmark(args) -> int:
    from .evolve import parse_score
    root = _project_root(args)
    state = State.load(root)
    config = _config(args)
    command = (state.plan.get("commands") or {}).get("benchmark")
    if not command:
        print("this project defines no benchmark command", file=sys.stderr)
        return 1
    log = for_workspace(root, config.get("logging.level"), args.quiet)
    with Sandbox(root, config.get("sandbox", {}), log) as sandbox:
        result = sandbox.run(command)
    print(result.summary())
    score = parse_score(result.stdout + result.stderr)
    print(f"\nscore: {score}")
    return 0 if result.ok else 1


def cmd_evolve(args) -> int:
    from .factory import Factory
    root = _project_root(args)
    factory = Factory(root, _config(args), quiet=args.quiet)
    if args.generations:
        factory.config.data["evolution"]["max_generations"] = args.generations
    with _server_for(factory), factory.sandbox:
        phases.optimization(factory)
    factory.save("evolve")
    metrics = factory.state.metrics.get("optimization")
    print(json.dumps(metrics, indent=2) if metrics else "no measurable objective")
    return 0


def cmd_eval(args) -> int:
    """Measure the factory itself against a fixed suite of ideas."""
    from .harness import SUITE, compare, run_suite, summarise, write_report
    if args.compare:
        print(compare(Path(args.compare[0]), Path(args.compare[1])))
        return 0
    if args.list:
        for task in SUITE:
            print(f"{task.name:<14} {task.idea}")
        return 0
    root = Path(args.into).resolve() if args.into else (
        Path.cwd() / "evaluations" / time.strftime("%Y%m%d-%H%M%S"))
    trials = run_suite(root, _config(args), names=args.task,
                       max_iterations=args.max_iterations, quiet=args.quiet,
                       on_trial=lambda trial: print(trial.row(), flush=True),
                       budget=args.budget * 60 if args.budget else None)
    print("\n" + summarise(trials))
    print(f"\nreport: {write_report(root, trials)}")
    return 0 if trials and all(t.completed for t in trials) else 2


def cmd_improve(args) -> int:
    """Section 35: improve one of Hermes' own components, under gates."""
    from .improve import Improvement, TARGETS
    from .log import NULL

    if args.list:
        for name, files in TARGETS.items():
            print(f"{name:<14} {', '.join(files)}")
        return 0
    if not args.target:
        print("name a component with --target, or list them with --list",
              file=sys.stderr)
        return 1
    root = Path(__file__).resolve().parent.parent
    if not (root / ".git").is_dir():
        print(f"{root} is not a git checkout of Hermes. Self-modification "
              "happens on a branch, so it needs one.", file=sys.stderr)
        return 1

    from contextlib import nullcontext
    proposal, agents = None, {}
    # A proposal read from a file needs no model, and therefore no server.
    server = nullcontext()
    if args.proposal:
        proposal = json.loads(Path(args.proposal).read_text())
    else:
        from .agents import build as build_agents
        from .models import Router
        from .server import running
        config = _config(args)
        router = Router(config, NULL)
        agents = build_agents(router, NULL)
        server = running(config, router, NULL)

    work = Improvement(root, agents, NULL)
    with server:
        record = work.attempt(args.target, proposal=proposal,
                              before=args.before, after=args.after)
    print(record.report())
    if args.promote and record.passed:
        print("promoted" if work.promote(record) else "promotion refused")
    if record.branch and not record.promoted:
        work.discard(record.branch)
    return 0 if record.passed else 2


def cmd_release(args) -> int:
    from .factory import Factory
    root = _project_root(args)
    factory = Factory(root, _config(args), quiet=args.quiet)
    if args.github:
        factory.github.enabled = True
    with _server_for(factory), factory.sandbox:
        phases.release(factory)
    factory.save("release")
    criteria = factory.state.metrics.get("completion", {})
    for name, ok in criteria.items():
        print(f"[{'pass' if ok else 'FAIL'}] {name}")
    return 0 if factory.state.phase == "COMPLETED" else 1


def cmd_checkpoint(args) -> int:
    root = _project_root(args)
    directory = root / ".hermes" / "checkpoints"
    if args.restore:
        target = directory / args.restore
        if not target.is_dir():
            print(f"no checkpoint {args.restore}", file=sys.stderr)
            return 1
        restore(root, target)
        print(f"restored {args.restore}")
        return 0
    if args.create:
        print(checkpoint(root, args.create).name)
        return 0
    for entry in sorted(p.name for p in directory.glob("*") if p.is_dir()):
        print(entry)
    return 0


def cmd_doctor(args) -> int:
    """Check that the local setup can actually run a project."""
    from .models import Router
    config = _config(args)
    ok = True
    print(f"config    {config.path or 'built-in defaults'}")
    resources = probe_resources()
    print(f"resources {resources}")
    from .server import unit_state
    unit = str(config.get("server.unit") or "")
    autostart = bool(config.get("server.autostart", True))
    state = unit_state(unit) if unit else "not configured"
    router = Router(config)
    # Resolved in its own pass, and before any endpoint is probed, so that a
    # config fault — an empty model name, a hosted base_url — can never be
    # excused below as a server that has merely not been started yet. Only
    # routed roles, so an unused role nothing would ever call cannot fail this.
    try:
        for task in router.routed_tasks():
            config.model_for(task)
    except ConfigError as exc:
        print(f"model     FAIL {exc}", file=sys.stderr)
        ok = False
    else:
        try:
            for line in router.preflight():
                print(f"model     {line}")
        except ConfigError as exc:
            if autostart and state in ("inactive", "activating", "deactivating"):
                # Not a fault: the run that needs the server is what starts
                # it, and between runs it is meant to be down.
                print(f"model     {unit} is {state}; the next run starts it")
            else:
                print(f"model     FAIL {exc}", file=sys.stderr)
                ok = False
    if unit:
        print(f"server    {unit} {state}"
              f"{'' if autostart else ' (autostart off: start it yourself)'}")
    # What a ticket that keeps failing will be asked next, and whether every
    # rung names a role that exists: a typo there is otherwise discovered on
    # the third attempt of some ticket in the middle of an unattended run.
    from .agents import PROMPTS
    from .implement import Ladder
    ladder = Ladder.from_config(config)
    for index, rung in enumerate(ladder.rungs, 1):
        attempt = f"{index}+" if index == len(ladder.rungs) else str(index)
        print(f"ladder    attempt {attempt}: {rung.candidates} sample(s), "
              f"{rung.files} files, {rung.role or 'implementer'}")
    for role in sorted({r.role for r in ladder.rungs if r.role} - set(PROMPTS)):
        print(f"ladder    WARN execution.escalation names no role {role!r}; "
              "that rung falls back to the implementer", file=sys.stderr)
    sandbox_cfg = config.get("sandbox", {})
    probe = Sandbox(Path.cwd(), sandbox_cfg)
    print(f"sandbox   runtime={probe.runtime} image={sandbox_cfg.get('image')} "
          f"network={sandbox_cfg.get('network')}")
    if probe.runtime != sandbox_cfg.get("runtime"):
        print("sandbox   WARN docker unavailable; falling back to local execution")
    import shutil as _shutil
    for tool in ("git", "gh", "docker"):
        print(f"tool      {tool}: {_shutil.which(tool) or 'not found'}")
    return 0 if ok else 1


def cmd_config(args) -> int:
    if args.sample:
        print(SAMPLE_CONFIG)
        return 0
    config = _config(args)
    print(f"# source: {config.path or 'built-in defaults'}")
    print(config.dump())
    return 0


# -- parser ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes",
        description="Autonomous software project factory driven by local models.")
    parser.add_argument("-p", "--project", help="project directory")
    parser.add_argument("-c", "--config", help="path to hermes.yaml")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress the log stream")
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new", help="create a project from an idea and build it")
    new.add_argument("idea")
    new.add_argument("--into", help="target directory (default projects/<slug>)")
    new.add_argument("--no-run", action="store_true", help="scaffold only")
    new.add_argument("--github", action="store_true", help="publish when complete")
    new.add_argument("--force", action="store_true")
    new.add_argument("-n", "--max-iterations", type=int)
    new.set_defaults(func=cmd_new)

    run = sub.add_parser("run", help="continue an existing project autonomously")
    run.add_argument("-n", "--max-iterations", type=int)
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="show current project state")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    sub.add_parser("pause", help="pause the loop after the current iteration").set_defaults(
        func=cmd_pause)

    resume = sub.add_parser("resume", help="unpause and continue")
    resume.add_argument("-n", "--max-iterations", type=int)
    resume.set_defaults(func=cmd_resume)

    logs = sub.add_parser("logs", help="show the machine-readable log")
    logs.add_argument("-n", "--lines", type=int, default=40)
    logs.add_argument("-f", "--follow", action="store_true")
    logs.add_argument("--json", action="store_true")
    logs.set_defaults(func=cmd_logs)

    inspect = sub.add_parser("inspect", help="show spec, plan, board and history")
    inspect.add_argument("-t", "--ticket", help="show one ticket in full")
    inspect.set_defaults(func=cmd_inspect)

    journal = sub.add_parser("journal", help="show what each agent did")
    journal.add_argument("-n", "--lines", type=int, default=20)
    journal.add_argument("-t", "--ticket", help="only this ticket's entries")
    journal.add_argument("--json", action="store_true")
    journal.set_defaults(func=cmd_journal)

    sub.add_parser("test", help="run the project's verification gates").set_defaults(
        func=cmd_test)
    sub.add_parser("benchmark", help="run the project's benchmark").set_defaults(
        func=cmd_benchmark)

    evolve = sub.add_parser("evolve", help="run evolutionary optimisation")
    evolve.add_argument("-g", "--generations", type=int)
    evolve.set_defaults(func=cmd_evolve)

    ev = sub.add_parser("eval", help="measure the factory against its own suite")
    ev.add_argument("-t", "--task", action="append", help="only this task (repeatable)")
    ev.add_argument("--into", help="directory to build the evaluation in")
    ev.add_argument("-n", "--max-iterations", type=int)
    ev.add_argument("--budget", type=float, metavar="MINUTES",
                    help="wall-clock ceiling per task (default 30)")
    ev.add_argument("--list", action="store_true", help="show the suite and exit")
    ev.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="compare two results.json files")
    ev.set_defaults(func=cmd_eval)

    imp = sub.add_parser("improve", help="improve Hermes itself, under gates")
    imp.add_argument("-t", "--target", help="component to improve")
    imp.add_argument("--list", action="store_true", help="show targets and exit")
    imp.add_argument("--proposal", metavar="FILE",
                     help="a change as JSON, instead of asking a model for one")
    imp.add_argument("--before", metavar="FILE",
                     help="results.json measured before the change")
    imp.add_argument("--after", metavar="FILE",
                     help="results.json measured after the change")
    imp.add_argument("--promote", action="store_true",
                     help="merge the candidate, but only if every gate passes")
    imp.set_defaults(func=cmd_improve)

    release = sub.add_parser("release", help="clean-room validate and release")
    release.add_argument("--github", action="store_true")
    release.set_defaults(func=cmd_release)

    ckpt = sub.add_parser("checkpoint", help="list, create or restore checkpoints")
    ckpt.add_argument("--create", metavar="LABEL")
    ckpt.add_argument("--restore", metavar="NAME")
    ckpt.set_defaults(func=cmd_checkpoint)

    sub.add_parser("doctor", help="verify models, sandbox and tooling").set_defaults(
        func=cmd_doctor)

    cfg = sub.add_parser("config", help="show effective configuration")
    cfg.add_argument("--sample", action="store_true", help="print a starter hermes.yaml")
    cfg.set_defaults(func=cmd_config)
    return parser


def _hpf_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpf",
        description="Build the project in this directory from a single idea, "
                    "autonomously, using locally installed models.",
        epilog='hpf "Build a ..."   start here    |    hpf   continue what is here')
    parser.add_argument("idea", nargs="*",
                        help="the project idea; quoting is optional")
    parser.add_argument("-n", "--max-iterations", type=int)
    parser.add_argument("-c", "--config", help="path to a hermes.yaml")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="suppress the log stream")
    parser.add_argument("--github", action="store_true", help="publish when complete")
    parser.add_argument("--force", action="store_true",
                        help="restart even though this directory holds a project")
    return parser


def hpf(argv: list[str] | None = None) -> int:
    """``hpf "idea"``: the whole factory, in the directory you are standing in.

    The same command starts and continues the work, because deciding which of
    those you meant is exactly the kind of question the factory is supposed to
    answer for itself: an idea starts a project here, no idea resumes the one
    already here.
    """
    parser = _hpf_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    here = Path.cwd()
    idea = " ".join(args.idea).strip()
    existing = (here / ".hermes" / "state" / "state.json").is_file()

    head = (["-q"] if args.quiet else []) + (["-c", args.config] if args.config else [])
    tail = ["-n", str(args.max_iterations)] if args.max_iterations else []

    if not idea:
        if not existing:
            parser.print_help(sys.stderr)
            print(f"\nno Hermes project in {here}; give it an idea to start one.",
                  file=sys.stderr)
            return 2
        return main([*head, "--project", str(here), "run", *tail])

    if existing and not args.force:
        print(f"{here} already holds a Hermes project. Run `hpf` with no idea to "
              "continue it, or pass --force to start over here.", file=sys.stderr)
        return 1

    return main([*head, "new", idea, "--into", str(here),
                 *(["--github"] if args.github else []),
                 *(["--force"] if args.force else []), *tail])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # Every record this invocation writes says which invocation wrote it.
        # A project's log file outlives any one command, and a one-off
        # `hermes test` lands in the same file as the loop that preceded it.
        with scope(run=uuid.uuid4().hex[:8], command=args.command):
            return args.func(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; state is checkpointed, resume with `hermes run`",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
