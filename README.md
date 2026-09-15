# Hermes Project Factory

An autonomous software factory. You give it one sentence; it researches,
specifies, plans, implements, tests, optimises, documents and packages a
working project without asking you anything else.

Every model call goes to a model already installed on your machine. There is
no path to a hosted API anywhere in the codebase, and a non-local endpoint is
rejected at configuration time.

```console
$ ./install.sh
$ mkdir -p ~/projects/json-diff && cd ~/projects/json-diff
$ hpf "Build a deterministic JSON diff tool in Python"
```

`hpf` builds the project in the directory you are standing in and runs to
completion on its own. Run `hpf` there again with no idea to continue an
interrupted project.

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
research round that changes nothing is treated as a failed round. Enable
`research.network` and the round also reads: the researcher names the pages
worth consulting, the sandbox fetches them, the bytes are kept under
`research/` as evidence, and a finding may cite only a page that was
actually retrieved.

**Planning** decomposes the architecture into tickets small enough for one
agent to implement, test and verify in a single pass. Dependency cycles are
detected and broken before the scheduler ever sees them.

**Implementation** picks the next ticket by score, not by order: priority,
how much other work it unblocks, expected information gain, cost, risk and
past failures all feed the choice. Each ticket is generated, applied,
verified, reviewed, and either completed or diagnosed and repaired. While one
ticket spends minutes on the CPU in its gates, the next independent one is
generated on the GPU — but only when the host actually has room for a second
concurrent model call, and never while a benchmark is being timed, because a
measurement that shares the machine records the contention rather than the
code.

**Optimisation** runs an AlphaEvolve-style loop when the project has an
automatically measurable objective. Correctness gates performance: a
candidate that fails the tests is discarded before its score is read.

**Release** exports a pristine copy of the tracked tree into a fresh sandbox
and builds and tests it there, proving the repository is reproducible from a
clean checkout rather than from leftover build state.

With GitHub enabled, the board is projected onto issues as tickets move: one
issue per ticket, closed when the ticket is, so a running project is
observable without shell access to the host. The projection is one way. The
board is the source of truth, an edit made on GitHub is never read back, and
each issue says so — a loop specified to run unattended should not have a
text box a human can steer it from.

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
- Optionally the OpenHands agent layer. Its CLI pins CPython 3.12, so
  `install.sh` gives it an environment of its own rather than sharing Hermes';
  `uv` provisions that interpreter for you if you have it.

Hermes itself needs one third-party package: PyYAML. OpenHands, when you
install it, keeps its own dependency tree out of Hermes' virtualenv.

## Install

```console
$ ./install.sh
```

It builds a virtualenv for the factory, installs it, links `hpf` and `hermes`
into `~/.local/bin`, writes a starter config to `~/.config/hermes/config.yaml`
if you have none, pre-pulls the sandbox image, installs the OpenHands agent
layer beside Hermes, and finishes by running `hermes doctor`. It installs no
model: Hermes is local-model-only and will not guess which one you want.

Nothing is installed for the Feynman research workflow, because it is part of
Hermes itself rather than a package.

| Flag | Effect |
|---|---|
| `--dev` | also install the test dependencies |
| `--system` | link into `/usr/local/bin` instead of `~/.local/bin` |
| `--bin DIR`, `--venv DIR` | choose where the commands and the virtualenv go |
| `--no-docker-image` | skip pre-pulling the sandbox image |
| `--no-config` | leave `~/.config/hermes/config.yaml` alone |
| `--no-openhands` | skip the optional OpenHands agent layer |

A virtualenv rather than a `--user` install, because most distributions now
mark the system interpreter externally managed and `pip` refuses to write to
it at all. The config in `$HOME` is what lets `hpf` work in project
directories that have no `hermes.yaml` of their own.

To install by hand instead:

```console
$ pip install -e .
$ hermes doctor
```

`hermes doctor` checks that every routed model is actually installed on your
local server, that the sandbox runtime works, and that the tooling is present.
It refuses to guess.

## Uninstall

```console
$ ./uninstall.sh
```

It removes the virtualenv, the `hpf` and `hermes` links, the editable-install
leftovers in this repository and the OpenHands agent layer. It keeps your
config, because the model routing in it is yours rather than something the
installer wrote, and it keeps the sandbox image, because `python:3.11-slim` is
a stock image other containers may share. A link on your PATH that points
somewhere other than the environments being removed is left alone.

| Flag | Effect |
|---|---|
| `--config` | also delete `~/.config/hermes/config.yaml` |
| `--docker-image` | also delete the pre-pulled sandbox image |
| `--all` | both of the above |
| `--keep-openhands` | leave the OpenHands agent layer installed |
| `--system`, `--bin DIR`, `--venv DIR` | mirror the install flags |
| `--dry-run` | report what would be removed, remove nothing |

Generated projects are never touched: they are ordinary directories the
installer never created. To finish, delete this repository.

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

Research fetches, where you enable them, are the one thing that reaches the
network, and they run in a one-shot container rather than the project's own.
Only public addresses are fetched: the host is resolved and the *address* is
judged, since a public hostname is free to point at `127.0.0.1`, and every
redirect hop is judged again. That is what stops a model-supplied URL from
reaching your inference server, your LAN, or a cloud metadata endpoint.

A static scan runs over generated sources on every ticket and again at
release, and it blocks completion regardless of whether the tests pass.

## Commands

| Command | What it does |
|---|---|
| `hpf "idea"` | build that project in the current directory, start to finish |
| `hpf` | continue the project in the current directory |
| `hermes new "idea"` | scaffold a project and build it autonomously |
| `hermes run` | continue an existing project |
| `hermes status` | phase, progress, active ticket, board |
| `hermes pause` / `hermes resume` | stop after the current iteration, and restart |
| `hermes logs -f` | follow the machine-readable log |
| `hermes inspect [-t T003]` | spec, plan, board, history, or one ticket |
| `hermes journal` | what each agent did, the record the agents themselves read |
| `hermes test` | run the project's verification gates |
| `hermes benchmark` | run the project's benchmark and report the score |
| `hermes evolve -g 30` | run evolutionary optimisation |
| `hermes eval` | build a fixed suite of ideas and report what the factory cost |
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
$ ./install.sh --dev
$ .venv/bin/pytest -q
```

The suite includes an end-to-end run of the entire state machine against a
stub model, so the loop's ability to reach `COMPLETED` unaided is checked on
every commit without needing inference.

## Measuring the factory

`hermes eval` builds a fixed suite of small projects and reports, per idea,
whether it reached `COMPLETED`, how many tickets completed versus were
abandoned, what share of tickets passed every gate on the first attempt, and
how the wall clock split between inference and the sandbox.

```console
$ hermes eval --into evaluations/before
$ hermes eval --into evaluations/after
$ hermes eval --compare evaluations/before/results.json \
                        evaluations/after/results.json
```

Each task also gets a wall-clock ceiling (`--budget MINUTES`, 30 by default),
because iterations bound how many phases run rather than how long one takes,
and without it a single stuck command lets one idea consume the whole run. A
task stopped that way is marked `TIME` and its numbers are reported as a floor
rather than a result.

The report says what it does not know. Wall clock that neither ledger explains
is printed as `unaccounted`, time both ledgers claim is printed as `overlap`
(prefetch generates while gates run, so the two may legitimately exceed the
clock), and a suite where every task passed identically, or none passed at
all, is labelled as having no resolution — real numbers that still cannot rank
one factory against another. `--compare` says the same thing about a pair of
runs where nothing moved.

This exists because everything else in this section is a trade. Constrained
decoding, search/replace edits, prefetching the next ticket, racing several
samples through the gates and the noise-aware evolution selector all cost
something to buy something, and which way a given change lands depends on the
model and the machine. The first-pass rate and the GPU/CPU split are the two
numbers that say which. It is also the precondition for the factory improving
its own components: a self-modification that cannot be measured cannot be
promoted.

## Limitations

- Output quality tracks the local model. A small model produces a small
  project.
- Agents patch files with exact search/replace edits and fall back to whole
  files. A local model that cannot copy an anchor character for character
  loses the edit and retries, which costs a generation.
- OpenHands is installed by default but stays dormant until you set
  `openhands.enabled: true`; without it Hermes uses its own implementer. The
  delegation path is checked against a live install: the invocation and the
  `LLM_*` variables are what OpenHands 1.21 accepts, and a run reaches only the
  endpoint Hermes names. What a full multi-file ticket costs there, against a
  real local model, is still unmeasured.
- `hermes eval` has only ever been run against the stub model in the test
  suite. Its bookkeeping, budgets and the honesty checks above are tested, but
  no suite run against a real local model has been recorded, so the eight
  tasks are uncalibrated: whether they can actually separate a better factory
  from a worse one on your model is exactly what the `resolution` line will
  tell you the first time you run it.
- Research works from what the model knows until you set
  `research.network: true`, which buys it real sources within a page, byte
  and time budget. The retrieval path is exercised against a real server on
  every commit; which URLs a given local model actually names, and how often
  they exist, is unmeasured.

## License

MIT
