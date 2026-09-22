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
`research.network` and the round also reads: a search turns each question
into URLs that exist, the researcher picks which of the results are worth
reading, the sandbox fetches those, the bytes are kept under `research/` as
evidence, and a finding may cite only a page that was actually retrieved.
What those pages say reaches the rest of the round as short cited claims and
never as prose, because a page is written by a stranger and the prompt it
lands in belongs to an agent that writes files — see [Isolation](#isolation).
The URLs come from a search rather than from the model's memory because a
local model asked to recall one mostly invents it, and an invented URL
spends a page of the budget to return a 404. Point `research.search` at a
SearXNG instance you host to keep the lookup off a third party. The page
budget belongs to the run, not to a round, so it still bounds the total when
a failing ticket escalates into research again. Those copies stay on your
machine: they are gitignored
and excluded from the release export, so a clone gets the citations and the
extracted text without republishing someone else's pages under the project's
licence.

**Experiments** are how a hypothesis stops being a note. A round that ends in
a testable claim queues it as an experiment ticket, and that ticket is run
rather than implemented: the Benchmark Engineer writes the smallest program
that measures the claim, the sandbox runs it, and a researcher shown both the
numbers and the program that produced them says what was settled. The verdict
is about the experiment, not the claim — a refuted hypothesis is a finished
experiment, and completes its ticket. Each one lands in its own directory
under `experiments/` with the program, every run's output and a `record.json`
carrying the hypothesis, parameters, configuration, code version, environment,
measurement and verdict, which is enough to repeat it without Hermes. Nothing
is overwritten: running the same hypothesis again gets the next identifier, so
the two results can be compared rather than one replacing the other. What was
measured goes back into the research log, which is what stops the next round
asking the question this one just answered.

**Planning** decomposes the architecture into tickets small enough for one
agent to implement, test and verify in a single pass. Dependency cycles are
detected and broken before the scheduler ever sees them.

**Implementation** picks the next ticket by score, not by order: priority,
how much other work it unblocks, expected information gain, cost, risk and
past failures all feed the choice. Each ticket is generated, applied,
verified, reviewed, and either completed or diagnosed and repaired. A repeated
failure is not retried identically — the same prompt at the same temperature
answers the same way — so each attempt is asked harder than the last: attempt
two races two samples and is shown more of the existing code, attempt three
goes to the reasoning role instead of the implementer, told to work out why
the approach cannot be made to work rather than resend it, and asked for the
whole file rather than a search/replace patch — the two formats fail in
different ways, and reconsidering an approach is the one thing a patch cannot
express. That last change applies only to the files the previous attempt
already failed on: rewriting the rest whole is how a large file loses the code
nobody meant to touch. The ladder is
`execution.escalation`, and it climbs roles rather than models, because
routing a retry to a second model would unload the resident one mid-ticket.

That ladder is a guess, and the factory checks it. Every attempt is recorded
in a ledger outside the workspace — the failure it was handed, how it was
asked, whether that worked — so the record outlives the project that produced
it. Once the same class of failure has been watched enough times, the rung
with the better measured record replaces the one the ladder named, and a rung
that escalates for no measured benefit is dropped for the cheaper one that
does as well. The policy is deliberately timid: it only overrules a rung it
has watched fail, it compares lower bounds rather than rates so two lucky
successes do not outrank forty observed ones, and with no evidence at all it
falls through to the fixed ladder. `hermes ledger` shows what has been
learned; `execution.ledger: false` turns it off.
What an agent is handed of the existing code is a guess — the ticket's
artifact list was written before any of it existed — so the implementer and
the reviewer can also look: `read_file`, `grep` and `list_dir` over the workspace, read-only,
under a budget of a few calls per answer. Checking a signature costs one
lookup; inventing one costs a whole generate, verify and triage cycle. While
one ticket spends minutes on the CPU in its gates, the next independent one is
generated on the GPU — but only when the host actually has room for a second
concurrent model call, and never while a benchmark is being timed, because a
measurement that shares the machine records the contention rather than the
code.

**Optimisation** runs an AlphaEvolve-style loop when the project has an
automatically measurable objective. Correctness gates performance: a
candidate that fails the tests is discarded before its score is read.
Candidates arrive as the same search/replace patches the implementer sends,
for a sharper version of the same reason: the targets are whichever files
dominate the benchmark's cost, which is to say the largest ones, and a
rewrite that drops a constant nothing tests still passes the gate and then
parents every generation after it. So a target is shown whole or not offered
at all — a file too big for the evolver's context is dropped from the target
list rather than truncated — and a patch that does not anchor is reported back
and re-asked rather than half-applied.

**Release** exports a pristine copy of the tracked tree into a fresh sandbox
and builds and tests it there, proving the repository is reproducible from a
clean checkout rather than from leftover build state.

With GitHub enabled, the repository is created as soon as planning produces a
board, and the board is projected onto issues as tickets move: one issue per
ticket, closed when the ticket is, so a running project is observable without
shell access to the host. The code itself waits for the first ticket to pass
its gates — repositories are private by default, and a scaffold nobody has
validated is not worth showing. The projection is one way. The
board is the source of truth, an edit made on GitHub is never read back, and
each issue says so — a loop specified to run unattended should not have a
text box a human can steer it from.

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU with current drivers. `install.sh` sets up the model server
  for you — vLLM, one model, port 6767 — and the published vLLM wheels are
  CUDA builds. Without a GPU the installer skips the server and leaves you to
  point `models.default` at a server you run yourself; Ollama, `llama.cpp` and
  any other OpenAI-compatible endpoint stay fully supported in config.
- VRAM enough for the model you choose. vLLM keeps the entire model resident
  and has no CPU offload, so an 8 GiB card means a 7B 4-bit model, not a 30B
  one. The default is a 7B AWQ coder at roughly 5 GiB.
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
- `uv` is also how vLLM gets a supported interpreter, since it has no wheels
  for the newest CPython releases yet.

Hermes itself needs one third-party package: PyYAML. vLLM and OpenHands each
get an environment of their own, so neither drags torch nor a hosted-provider
SDK into the factory's virtualenv.

## Install

```console
$ ./install.sh
```

It builds a virtualenv for the factory, installs it, links `hpf` and `hermes`
into `~/.local/bin`, pre-pulls the sandbox image, installs the OpenHands agent
layer beside Hermes, sets up the vLLM model server described below, writes a
config matching the model it is serving, and finishes by running
`hermes doctor`. The server is started once so that the install proves the
model loads on this GPU, then stopped again: from there on Hermes starts it
when a project starts and stops it when the run ends.

It asks exactly one question, once: which model to serve. Either a Hugging
Face repository to download and where to keep the weights, or a directory of
weights you already have. Nothing else is asked, and nothing is downloaded
behind your back — run it without a terminal and it tells you which flag to
pass rather than fetching gigabytes unattended.

The slow steps report progress. Where the underlying tool already prints real
byte counts — `uv` and `hf download`, which together account for most of the
wait — that output is left to speak for itself; the silent steps get a
spinner with elapsed time, and the wait for vLLM to load the weights gets a
bar against its timeout. Piped to a file it degrades to one line per step
instead of a megabyte of carriage returns.

Nothing is installed for the Feynman research workflow, because it is part of
Hermes itself rather than a package.

| Flag | Effect |
|---|---|
| `--dev` | also install the test dependencies |
| `--system` | link into `/usr/local/bin` instead of `~/.local/bin` |
| `--bin DIR`, `--venv DIR` | choose where the commands and the virtualenv go |
| `--no-docker-image` | skip pre-pulling the sandbox image |
| `--no-config` | leave `~/.config/hermes/config.yaml` alone |
| `--force-config` | rewrite an existing config to point at this server |
| `--no-openhands` | skip the optional OpenHands agent layer |
| `--model REPO` | Hugging Face repository to download and serve |
| `--model-path DIR` | serve a directory of weights you already have |
| `--models-dir DIR` | where downloaded weights are stored |
| `--port N` | port vLLM listens on (default `6767`) |
| `--max-model-len N` | context vLLM serves (default `16384`) |
| `--gpu-util F` | fraction of VRAM vLLM may use (default `0.90`) |
| `--no-server` | install nothing for vLLM and start no server |
| `--boot` | also run the server at boot, not just during a run |

## Model server

The factory talks to one vLLM server on `http://localhost:6767/v1`, installed
as a **systemd user unit** called `hermes-vllm.service`. A unit rather than a
command you remember to type, because something has to own restarts across a
run that lasts hours.

The unit is installed but **not enabled**. vLLM pins the whole model in VRAM
for as long as it lives — it has no CPU offload and never evicts — so a server
running since boot is gigabytes of GPU given to an idle process. Hermes
instead starts the unit itself when a project starts or continues, waits for
the weights to load, and stops it when the run ends. A server you started by
hand is left running, because you did that for a reason.

```console
$ systemctl --user status hermes-vllm.service
$ journalctl --user -u hermes-vllm.service -f
$ systemctl --user start hermes-vllm.service    # bring it up by hand
```

`hermes doctor` reports an installed-but-stopped unit as `inactive` and does
not count it against the setup, since the next run brings it up. A config
fault still fails: an empty model name or a hosted `base_url` is not excused
by a server that happens to be down.

```yaml
server:
  autostart: true              # start the unit when a run begins
  unit: hermes-vllm.service
  wait: 600                    # seconds allowed for the weights to load
  stop_after_run: true         # false keeps the model resident between runs
```

Set `stop_after_run: false` to pay the load once and keep the model in VRAM,
which is also what you want when two projects share one server — Hermes only
ever stops a server it started itself, but it cannot tell that another run is
still using one. `./install.sh --boot` restores the always-up arrangement:
the unit is enabled, and lingering (`loginctl enable-linger`) is turned on,
which is the part that makes "at boot" true rather than "at next login" for a
user unit.

A user unit, not a system one, so nothing runs as root and the weights stay
readable by you alone.

Serving is offline: the unit sets `HF_HUB_OFFLINE=1` and points vLLM at a
directory on disk, so the server never reaches the network once the weights
are downloaded. The name vLLM answers to is the weights directory's name, and
the config is written with that same string, because `hermes doctor` compares
what is configured against what `/v1/models` reports.

To change model later, re-run `./install.sh --model REPO` (or `--model-path`),
or edit the unit and `systemctl --user restart` it. Keep `context` in the
config equal to the server's `--max-model-len`: vLLM reserves the KV cache for
the full context at startup, which makes it a VRAM commitment rather than a
ceiling, and it is the usual reason a server installs cleanly and then fails
to load.

A virtualenv rather than a `--user` install, because most distributions now
mark the system interpreter externally managed and `pip` refuses to write to
it at all. The config in `$HOME` is what lets `hpf` work in project
directories that have no `hermes.yaml` of their own.

To install by hand instead:

```console
$ pip install -e .
$ hermes doctor
```

That installs the factory only — no model server — so you would be pointing
`models.default` at a server you start yourself.

`hermes doctor` checks that every routed model is actually installed on your
local server, that the sandbox runtime works, and that the tooling is present.
It refuses to guess.

## Uninstall

```console
$ ./uninstall.sh
```

It stops and removes the `hermes-vllm.service` unit, the virtualenvs for the
factory, vLLM and OpenHands, the `hpf` and `hermes` links, and the
editable-install leftovers in this repository. It keeps the sandbox image,
because `python:3.11-slim` is a stock image other containers may share, and it
keeps the downloaded weights, because those are gigabytes you would have to
fetch again. A link on your PATH that points
somewhere other than the environments being removed is left alone, and
lingering is left enabled, since other user services of yours may now rely on
it.

Your config is kept — the routing in it is yours — but the endpoint of the
removed server is retired in place: the model name is emptied and the dead
`base_url` is marked as such, so Hermes fails with "no model name/path
configured" rather than advising you to start a server that no longer exists.
The original is saved as `config.yaml.bak`. A config pointing somewhere other
than a local vLLM is left untouched.

| Flag | Effect |
|---|---|
| `--config` | delete `~/.config/hermes/config.yaml` instead of retiring the endpoint in it |
| `--docker-image` | also delete the pre-pulled sandbox image |
| `--models` | also delete the downloaded model weights |
| `--all` | `--config` and `--docker-image`, but not `--models` |
| `--keep-openhands` | leave the OpenHands agent layer installed |
| `--keep-server` | leave the vLLM unit and environment installed |
| `--system`, `--bin DIR`, `--venv DIR` | mirror the install flags |
| `--dry-run` | report what would be removed, remove nothing |

Generated projects are never touched: they are ordinary directories the
installer never created. To finish, delete this repository.

## Configuration

Hermes reads `hermes.yaml` from the working directory, then
`$XDG_CONFIG_HOME/hermes/config.yaml` if that variable is set, then
`~/.config/hermes/config.yaml` — the same file the installer writes. Print a
starter file with `hermes config --sample`.

```yaml
models:
  default:
    backend: vllm                    # vllm | openai | llamacpp | ollama
    base_url: http://localhost:6767/v1
    model: qwen2.5-coder-7b-instruct-awq
    context: 16384                   # must match the server's --max-model-len
  reasoner:
    backend: ollama                  # a second server, if you run one
    base_url: http://localhost:11434
    model: qwen3:30b-a3b

routing:
  architecture: reasoner
  research: reasoner
```

Routing every role to one model is usually right on a single GPU: two models
that are not co-resident turn every role switch into a full model load, and
the loop switches roles several times per ticket. `hermes doctor` warns when
the config asks for more than one.

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
Agent lookups go through the same check in the other direction: they read
only, only inside the workspace, and a path that escapes it comes back to the
model as a refusal it can act on rather than an error that fails the ticket.

Research fetches, where you enable them, are the one thing that reaches the
network, and they run in a one-shot container rather than the project's own.
Only public addresses are fetched: the host is resolved and the *address* is
judged, since a public hostname is free to point at `127.0.0.1`, and every
redirect hop is judged again. That is what stops a model-supplied URL from
reaching your inference server, your LAN, or a cloud metadata endpoint.

A fetched page is also a stranger writing into a prompt, so it is treated as
untrusted for the whole of its life. Nothing about that defence is a sentence
in a prompt, because any instruction written above a page can be argued with
by the page. Instead: retrieved text is read on one call that has the question
and the page and nothing else — no project context and no tools, since a
prompt carrying untrusted text cannot be given a toolbox at all — and what
that call may return is cut down in Python to a few short claims, each pinned
to a page that was really fetched. Every later prompt sees the claims, fenced
and labelled, never the page. The database records which pages were read and
why, never what they said, so the `recall` tool cannot hand a page to an agent
that also holds a shell. Finally, a research round that read pages may not
install anything, and any command it proposes must clear the security gate and
the classifier before it can reach a ticket; what is refused is written into
that round's notes under `docs/research/`.

A static scan runs over generated sources on every ticket and again at
release, and it blocks completion regardless of whether the tests pass.

That scan is a pattern list, which means it reads what it was told to look
for and nothing else. You can put a second opinion behind it: a local text
classifier that scores each chunk of generated source against a small
vulnerability taxonomy, for the defects no regex expresses. It is off by
default, and when on it is advisory — a finding is written into the gate's
detail, where the reviewer and the triager both read it, and the expensive
model decides whether it was right. The patterns keep their veto either way,
and a classifier that is down or unreachable leaves the gate exactly as it
was.

```yaml
security:
  classifier:
    enabled: true
    base_url: http://localhost:6768
    threshold: 0.75      # calibrate the checkpoint before this means anything
    block: false         # let a finding fail the gate, not merely warn
```

`contrib/laya_server.py` is a reference adapter serving that endpoint over
[Laya](https://huggingface.co/convaiinnovations/laya). It is a separate
process on purpose: the classifier wants torch and Hermes wants no
dependency beyond PyYAML. Note that an un-tuned checkpoint scores near chance
on a task it was not trained for — this is worth turning on once you have
fine-tuned one on labelled vulnerability data and measured its false positive
rate against your own code, and `block` is worth setting only after that.

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
| `hermes ledger` | which retry actually repaired which failure, across every project |
| `hermes context catch-up <reader>` | what has changed since that reader last looked, and only that |
| `hermes test` | run the project's verification gates |
| `hermes benchmark` | run the project's benchmark and report the score |
| `hermes evolve -g 30` | run evolutionary optimisation |
| `hermes eval` | build a fixed suite of ideas and report what the factory cost |
| `hermes improve --target planner` | improve Hermes itself, under gates |
| `hermes release --github` | clean-room validate, tag, and publish |
| `hermes checkpoint` | list, create or restore a checkpoint |
| `hermes doctor` | verify models, sandbox and tooling |

## Generated project layout

```
project/
├── CONTEXT.md          always-current orientation
├── AGENTS.md           rules for whoever works on this next
├── SPEC.md  PLAN.md    what and how
├── src/  tests/  benchmarks/
├── experiments/        one directory per experiment: program, runs, record
├── docs/
│   ├── architecture/  research/  decisions/  generated/
├── tickets/            kanban board, mirrored as markdown
├── research/  artifacts/
└── .hermes/            state, logs, metrics, evaluations, checkpoints
    └── context.db      what every agent recorded, queryable and incremental
```

Everything outside `.hermes/` is meant to be readable without Hermes. The
released repository excludes factory internals entirely.

Inside `.hermes/` sits `context.db`, the project's context database. Agents
are stateless, so everything one of them learned has to reach the next one
through the workspace, and a file costs the same to hand over whether or not
it changed. The database makes that handover incremental: records (a gate
result, a decision, a ticket's outcome, a remembered fact) and the project's
own files, split at their own boundaries and keyed by content hash, each
stamped with a rising revision number. A reader remembers the revision it last
saw and asks for what came after, so a one-function change in a 900-line file
arrives as one chunk rather than the file. It is SQLite and nothing but the
standard library, and it is derived, not authoritative — the files are on disk
and the journal is still JSONL, so `hermes context sync` rebuilds it from a
clean checkout and it is never committed.

A generated project's own build steps write to it too:

```sh
cargo test 2>&1 | hermes context log "cargo test" --kind gate
hermes context catch-up implementer   # what the next agent has not seen yet
hermes context search "retry backoff"
hermes context outline src/engine.rs  # what a file contains, without its body
```

The JSONL log under `.hermes/logs/` stamps every record with the work that
produced it — `run`, `phase`, `ticket` and `attempt` — so a line written deep
inside a model call or a sandbox command can be traced back to the ticket
attempt it belongs to. `run` identifies one `hermes` invocation, which is what
separates a resumed project's records from the earlier run's in the same file.

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
decoding, search/replace edits, workspace lookups, prefetching the next
ticket, racing several samples through the gates, escalating a repeated
failure and the noise-aware evolution selector all cost something to buy
something, and which way a given change lands depends on the model and the
machine. Each is a knob, so each can be measured the same way —
`execution.tool_calls: 0` in one run and the default in the other is the
comparison that says what lookups are worth on your model, and
`execution.escalation: []` against the default says whether asking harder
turns abandoned tickets into completed ones or only spends more GPU. The first-pass rate and the GPU/CPU split are the two
numbers that say which. It is also the precondition for the factory improving
its own components: a self-modification that cannot be measured cannot be
promoted.

## Improving the factory

`hermes improve --target planner` lets Hermes change one of its own
components. Section 35 permits that only under conditions, and each one is
enforced here rather than asked for in a prompt:

- The candidate is built as a git worktree on its own branch. The checkout
  Hermes is running from is never switched, so a candidate cannot rewrite the
  source of the process that is evaluating it.
- A change touching the test suite, the evaluation harness or the promotion
  gate itself is refused before it is evaluated, and the trusted copies are
  restored into the worktree before the suite runs. The cheapest way to pass
  a suite is to delete it, and a loop able to do that eventually will.
- Promotion needs the Hermes suite to pass *and* a measured benefit — `hermes
  eval` on each side, handed over as `--before` and `--after`. Without that
  evidence the gate refuses, so on a machine with no local model nothing is
  promotable. That is the correct answer rather than an obstacle.
- A promoted change is merged and then the suite is run again on the live
  tree, because a change can pass alone and fail in company. A failure there
  resets it.

Benchmark infrastructure is deliberately absent from the target list: it
lives in the harness, which is protected, so the factory cannot rewrite the
instrument it is judged by. None of this runs inside the autonomous project
loop — a factory building someone's project has no business rewriting itself
halfway through.

## Limitations

- Output quality tracks the local model. A small model produces a small
  project.
- Agents patch files with search/replace edits and fall back to whole files.
  An anchor whose indentation or trailing space drifted is re-located against
  the file and the replacement re-indented to match, so the common near-miss no
  longer costs a generation. What is never guessed is *which* of two possible
  sites was meant: an ambiguous anchor is refused unless the edit names the
  function or class it belongs to, because a wrong edit that still passes the
  tests is worse than a retry.
- Optimisation needs its targets to fit the evolver's context whole. A project
  whose hot file is larger than that gets no optimisation rather than a
  half-informed one; the log names the file and the room it missed by, and the
  answer is a bigger `context` or a benchmark that concentrates on less code.
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
  `research.network: true`, which buys it real sources within a page, search,
  byte and time budget. The search and retrieval path is exercised against a
  real server on every commit; how well a given local model turns a question
  into search terms, and how often the results answer it, is unmeasured. The
  default endpoint is DuckDuckGo's HTML interface, which is a third party and
  rate-limits: `research.search` takes any URL template, so a self-hosted
  SearXNG is one line of configuration.
- The fencing around retrieved pages is structural rather than advisory, and
  it is still not total. A page that persuades the researcher of something
  false has changed what the project believes, and the mechanism that stops
  the *wrong answer* is the one that was always there: a finding carries its
  confidence and its citation, and a claim has to survive the synthesis and
  the reviewer. Screening bites on the executable surface — the commands a
  research round proposes — while a dependency named in prose can still reach
  the architect and end up in the project's setup command, since the architect
  is where dependencies are legitimately chosen and no gate can tell a good
  package name from a bad one.

## License

MIT
