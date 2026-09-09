# Hermes Project Factory

An autonomous software factory. You give it one sentence; it researches,
specifies, plans, implements, tests, optimises, documents and packages a
working project without asking you anything else.

Every model call goes to a model already installed on your machine. There is
no path to a hosted API anywhere in the codebase, and a non-local endpoint is
rejected at configuration time.

```console
$ hermes new "Build a deterministic JSON diff tool in Python"
```

That command scaffolds a workspace, then runs to completion on its own.

## How it works

Hermes moves a project through an explicit state machine, persisting every
transition so an interrupted run resumes exactly where it stopped.

```
INTAKE -> RESEARCH -> SPECIFICATION -> PLANNING -> IMPLEMENTATION
       -> VALIDATION -> OPTIMIZATION -> DOCUMENTATION -> RELEASE -> COMPLETED
```

**Intake** turns the idea into a formal specification, inferring defaults for
everything you did not say and recording each inference as an assumption.

**Research** runs a Feynman loop only when the problem warrants it: explain
the problem plainly, name the unknowns, attack them, challenge the answers,
then convert what survives into decision records, hypotheses and tickets. A
research round that changes nothing is treated as a failed round.

**Planning** decomposes the architecture into tickets small enough for one
agent to implement, test and verify in a single pass. Dependency cycles are
detected and broken before the scheduler ever sees them.

**Implementation** picks the next ticket by score, not by order: priority,
how much other work it unblocks, expected information gain, cost, risk and
past failures all feed the choice. Each ticket is generated, applied,
verified, reviewed, and either completed or diagnosed and repaired.

**Optimisation** runs an AlphaEvolve-style loop when the project has an
automatically measurable objective. Correctness gates performance: a
candidate that fails the tests is discarded before its score is read.

**Release** exports a pristine copy of the tracked tree into a fresh sandbox
and builds and tests it there, proving the repository is reproducible from a
clean checkout rather than from leftover build state.

## Requirements

- Python 3.10 or newer
- A local inference server: Ollama, or anything OpenAI-compatible such as
  `llama.cpp`'s server or vLLM
- Docker, strongly recommended. Without it Hermes falls back to local
  subprocess execution, which is a much weaker isolation guarantee.
- A project directory the Docker engine can actually bind-mount. Under Docker
  Desktop on WSL a path such as `/tmp/project` mounts as an empty directory,
  because that path also exists inside the engine's own VM. Hermes checks this
  at startup and refuses to run blind, falling back to local execution with a
  warning; keep projects under your home directory to avoid it.
- `git`, and `gh` if you want GitHub publishing

Only one third-party package is needed: PyYAML.

```console
$ pip install -e .
$ hermes doctor
```

`hermes doctor` checks that every routed model is actually installed on your
local server, that the sandbox runtime works, and that the tooling is present.
It refuses to guess.

## Configuration

Hermes reads `hermes.yaml` from the working directory, or
`~/.config/hermes/config.yaml`. Print a starter file with
`hermes config --sample`.

```yaml
models:
  default:
    backend: ollama
    base_url: http://localhost:11434
    model: qwen3:30b-a3b
  coder:
    backend: openai
    base_url: http://localhost:8080/v1
    model: qwen2.5-coder-32b

routing:
  coding: coder
  evolution: coder
```

Any endpoint that is not loopback or a private address is refused unless you
set `allow_remote: true` on that model, which exists for a server you host
yourself on your own network.

## Isolation

Generated code is treated as hostile. It runs in a container with no network,
capped memory, CPU and process count, all capabilities dropped, and only the
project workspace mounted. Agent file writes are checked against path escapes
and a protected-path list, so generated code cannot reach Hermes state, the
git directory, or anything on the host. In the local fallback runtime the
environment is stripped of anything resembling a key, token or credential.

A static scan runs over generated sources on every ticket and again at
release, and it blocks completion regardless of whether the tests pass.

## Commands

| Command | What it does |
|---|---|
| `hermes new "idea"` | scaffold a project and build it autonomously |
| `hermes run` | continue an existing project |
| `hermes status` | phase, progress, active ticket, board |
| `hermes pause` / `hermes resume` | stop after the current iteration, and restart |
| `hermes logs -f` | follow the machine-readable log |
| `hermes inspect [-t T003]` | spec, plan, board, history, or one ticket |
| `hermes test` | run the project's verification gates |
| `hermes benchmark` | run the project's benchmark and report the score |
| `hermes evolve -g 30` | run evolutionary optimisation |
| `hermes release --github` | clean-room validate, tag, and publish |
| `hermes checkpoint` | list, create or restore a checkpoint |
| `hermes doctor` | verify models, sandbox and tooling |

## Generated project layout

```
project/
├── CONTEXT.md          always-current orientation
├── AGENTS.md           rules for whoever works on this next
├── SPEC.md  PLAN.md    what and how
├── src/  tests/  benchmarks/  experiments/
├── docs/
│   ├── architecture/  research/  decisions/  generated/
├── tickets/            kanban board, mirrored as markdown
├── research/  artifacts/
└── .hermes/            state, logs, metrics, evaluations, checkpoints
```

Everything outside `.hermes/` is meant to be readable without Hermes. The
released repository excludes factory internals entirely.

## Completion

A project is not finished because its tickets are closed. Hermes requires all
of: requirements satisfied, build succeeds, tests pass, no critical defects,
performance objective met where one exists, documentation present, the clean
room reproduces it, and the security scan is clean. Failing that, it goes
back to implementation, and after repeated failure it stops and says so
rather than declaring success.

## Development

```console
$ pytest -q
```

The suite includes an end-to-end run of the entire state machine against a
stub model, so the loop's ability to reach `COMPLETED` unaided is checked on
every commit without needing inference.

## Limitations

- Output quality tracks the local model. A small model produces a small
  project.
- Agents return complete files rather than diffs. That is reliable with local
  models but costs tokens on large files.
- OpenHands integration is optional and off by default; without it Hermes
  uses its own implementer.
- Research works from what the model knows. Network research is not enabled
  by default.

## License

MIT
