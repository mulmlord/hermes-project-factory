# Hermes Project Factory — Implementation Specification

## 1. Mission

Build **Hermes Project Factory**, a fully autonomous software-project factory that takes a high-level project idea as its only required user input and autonomously researches, plans, implements, tests, evaluates, improves, documents, and packages the resulting project.

The factory must not require user interaction after the initial idea prompt.

Hermes is the orchestration layer. It creates and manages an isolated project workspace, decomposes work into tickets, assigns work to locally installed AI models, validates every result, iterates until the project satisfies its acceptance criteria, and finally produces a clean release-ready repository.

The factory must support difficult engineering and research projects, including projects where the optimal implementation is not known in advance.

---

## 2. Core Constraints

### 2.1 Local-model-only requirement

Hermes supports **local models only**.

Do not depend on:

- Anthropic API
- OpenAI API
- Google Gemini API
- OpenRouter
- hosted inference APIs
- any other external commercial LLM API

The models used by Hermes must already be installed locally.

The architecture should support local inference backends such as:

- llama.cpp
- vLLM
- Ollama
- other locally hosted OpenAI-compatible or native inference servers

The model backend must be configurable.

### 2.2 Full autonomy

After receiving the initial project idea, Hermes must continue autonomously.

It must not ask the user:

- what to implement next
- which library to choose
- which ticket to work on
- whether a failed test should be retried
- whether research should continue
- whether the project is finished
- whether documentation should be generated

The system should make these decisions itself.

Human intervention may be supported as an optional emergency/debug mode, but it must never be required for the normal factory workflow.

### 2.3 Isolation

Every generated project must run inside an isolated environment.

Preferred architecture:

- Docker container
- disposable sandbox/worktree
- restricted filesystem access
- restricted network access
- resource limits
- project-local dependencies

The factory must protect the host machine from generated code.

Generated code must not be able to arbitrarily modify the Hermes installation or host filesystem.

### 2.4 No third-party software

Hermes and vLLM are the only software the factory installs or depends on.

When third-party software would be beneficial, it must not be added. Its function is reverse-engineered — from documentation and observable behaviour, never by copying its code — and implemented natively inside Hermes, under the same gates as any other change.

Docker, git and Python itself remain host prerequisites for now, because removing them without a native replacement would break isolation or version control. They are the first items on the reverse-engineering backlog, to be replaced once the factory has proved itself on real projects, as the first step toward a fully dependency-free factory.

This rule applies to Hermes itself. A generated project's own toolchain is chosen by that project's architect and installed inside that project's sandbox.

---

# 3. High-Level Architecture

Hermes should be composed of the following subsystems:

```text
User Idea
   |
   v
Project Intake
   |
   v
Research & Specification
   |
   v
Project Planner
   |
   v
Kanban / Ticket Engine
   |
   +-----------------------------+
   |                             |
   v                             |
Research Agents                  |
   |                             |
   v                             |
Implementation Agents            |
   |                             |
   v                             |
Testing / Verification           |
   |                             |
   v                             |
Evaluation                       |
   |                             |
   v                             |
Self-Improvement Loop -----------+
   |
   v
Release Validation
   |
   v
Documentation
   |
   v
Git Repository
   |
   v
Optional GitHub Push
```

---

# 4. Project Workspace

For every project Hermes must create a dedicated workspace.

Recommended structure:

```text
project/
├── CONTEXT.md
├── AGENTS.md
├── SPEC.md
├── PLAN.md
├── README.md
├── CHANGELOG.md
├── .gitignore
│
├── docs/
│   ├── architecture/
│   ├── research/
│   ├── decisions/
│   └── generated/
│
├── src/
├── tests/
├── benchmarks/
├── experiments/
│
├── tickets/
│   ├── backlog/
│   ├── active/
│   ├── blocked/
│   ├── review/
│   └── completed/
│
├── research/
├── artifacts/
└── .hermes/
    ├── state/
    ├── logs/
    ├── metrics/
    ├── evaluations/
    └── checkpoints/
```

Hermes must maintain machine-readable state in `.hermes/state`.

Human-readable project context must remain available through:

- `CONTEXT.md`
- `AGENTS.md`
- `SPEC.md`
- `PLAN.md`

---

# 5. Initial Project Intake

The user provides a project idea.

Example:

```text
Build an autonomous quantitative trading engine in C++.
```

Hermes must transform this into a formal project definition.

The intake stage should determine:

- objective
- expected deliverables
- constraints
- target platforms
- functional requirements
- non-functional requirements
- likely risks
- unknowns
- measurable success criteria

If information is missing, Hermes should infer reasonable defaults instead of asking the user.

All assumptions must be recorded.

---

# 6. Research Phase

Before implementation, Hermes should determine what is already known and what must be discovered.

Research can include:

- documentation
- academic literature
- existing open-source implementations
- algorithms
- benchmarks
- architecture patterns
- security considerations
- compatibility requirements
- implementation tradeoffs

Research results must be stored under:

```text
research/
```

and summarized in:

```text
docs/research/
```

Every major architectural decision should have a corresponding decision record.

---

# 7. Feynman Research Workflow

Hermes must integrate the **Feynman-style research workflow** for difficult problems.

Use the workflow when:

- the problem is poorly understood
- multiple competing approaches exist
- the optimal method is uncertain
- research is required to discover an efficient solution
- implementation requires non-obvious scientific or engineering reasoning

The workflow should follow:

```text
Problem
  |
  v
Explain the problem simply
  |
  v
Identify unknowns
  |
  v
Research unknowns
  |
  v
Construct candidate explanations
  |
  v
Challenge assumptions
  |
  v
Compare alternatives
  |
  v
Build minimal experiment
  |
  v
Measure
  |
  v
Update understanding
  |
  v
Repeat
```

Hermes should not merely summarize papers.

It must extract actionable knowledge and convert research into:

- hypotheses
- experiments
- implementation decisions
- benchmarks
- tickets

---

# 8. AlphaEvolve Integration

Hermes must integrate an **AlphaEvolve-style evolutionary optimization workflow**.

AlphaEvolve should be used when a project contains an objective that can be automatically measured.

Examples:

- runtime
- memory usage
- throughput
- benchmark score
- numerical accuracy
- compression ratio
- trading backtest metrics
- algorithmic efficiency
- compiler performance
- scheduling quality

The workflow should be:

```text
Baseline
   |
   v
Define objective
   |
   v
Generate candidate
   |
   v
Compile
   |
   v
Run tests
   |
   v
Run benchmark
   |
   v
Evaluate
   |
   +---- worse ----> discard
   |
   +---- better ---> keep
                       |
                       v
                  next generation
```

Candidates must be isolated.

Every candidate must pass correctness tests before its performance is considered.

Maintain:

- parent candidate
- mutation/change
- objective score
- correctness result
- benchmark results
- generation number
- artifact/checkpoint

Never optimize an implementation solely on a performance metric if correctness is not verified.

---

# 9. Where AlphaEvolve Is Used

AlphaEvolve-style optimization should primarily operate after a working baseline exists.

Recommended stages:

### Stage A — Baseline

Build a correct reference implementation.

### Stage B — Benchmark

Create deterministic evaluation functions.

### Stage C — Candidate generation

Use local models to propose code modifications.

### Stage D — Validation

Run:

1. compilation
2. unit tests
3. integration tests
4. correctness checks
5. benchmark

### Stage E — Selection

Only candidates that remain correct can become parents.

### Stage F — Evolution

Repeat until:

- target performance is reached
- improvement plateaus
- budget is exhausted
- maximum generations are reached

The evolution engine should support:

- single-objective optimization
- multi-objective optimization
- Pareto-front tracking
- regression detection
- rollback

---

# 10. OpenHands Integration

Hermes should support **OpenHands** as an optional execution/agent layer.

OpenHands can be used for tasks that benefit from a full software-engineering agent environment, including:

- repository exploration
- multi-file implementation
- debugging
- test-driven development
- refactoring
- issue resolution
- terminal interaction
- iterative implementation

OpenHands must operate inside the same isolated project environment.

Hermes remains the top-level orchestrator.

Conceptually:

```text
Hermes
   |
   +--> Local LLM
   |
   +--> OpenHands Agent
   |
   +--> Research Agent
   |
   +--> AlphaEvolve Engine
   |
   +--> Test/Evaluation System
```

OpenHands must not introduce a requirement for hosted model APIs.

Its model configuration must point to locally installed models/inference endpoints.

If OpenHands is unavailable, Hermes must continue using its native agent implementation.

**Status (§2.4):** OpenHands is no longer installed or invoked. The capabilities listed above are to be provided by Hermes' native agent, by reimplementing OpenHands' function rather than integrating it.

---

# 11. Agent System

Hermes should implement specialized agent roles.

Recommended roles:

### Architect

Responsible for:

- architecture
- interfaces
- technology choices
- system decomposition

### Researcher

Responsible for:

- literature
- documentation
- existing solutions
- technical investigation

### Implementer

Responsible for:

- writing code
- implementing tickets
- integrating components

### Tester

Responsible for:

- tests
- edge cases
- regression testing

### Reviewer

Responsible for:

- code review
- correctness
- maintainability
- security

### Benchmark Engineer

Responsible for:

- benchmarks
- performance measurements
- profiling

### Evolution Agent

Responsible for:

- AlphaEvolve-style optimization
- candidate generation
- mutation strategies
- performance improvement

### Documentation Agent

Responsible for:

- README
- API documentation
- architecture documentation
- usage examples

Agents should be stateless where possible and derive their working context from project state.

---

# 12. Kanban Ticket System

Hermes must manage work using a Kanban-style ticket system.

Tickets should move through:

```text
BACKLOG
   |
   v
READY
   |
   v
ACTIVE
   |
   v
REVIEW
   |
   +---- failed ----> ACTIVE
   |
   +---- passed ----> COMPLETED
```

Additional state:

```text
BLOCKED
```

Each ticket should contain:

```text
ID
Title
Objective
Context
Dependencies
Acceptance Criteria
Validation Commands
Expected Artifacts
Priority
Risk
Status
```

Hermes decides which ticket to execute next.

Ticket selection should consider:

- dependency graph
- priority
- risk
- expected information gain
- implementation cost
- project critical path

---

# 13. Ticket Generation

The planner must continuously decompose large objectives into smaller executable tasks.

A ticket should be small enough that an agent can:

1. understand it
2. implement it
3. test it
4. verify it

Large tickets should automatically be decomposed.

The system should continue decomposing until tasks become actionable.

---

# 14. Autonomous Execution Loop

The core Hermes loop should resemble:

```text
while project_not_complete:

    load_state()

    inspect_project()

    identify_next_best_action()

    if knowledge_gap:
        research()

    elif experiment_needed:
        run_experiment()

    elif implementation_needed:
        implement()

    elif optimization_possible:
        evolve()

    elif validation_needed:
        test()

    elif documentation_needed:
        document()

    update_state()

    evaluate_project()

    recover_from_failures()

    checkpoint()
```

No user interaction should be required.

---

# 15. Verification

Every implementation must be verified.

Verification should happen at multiple levels.

### Static

- compilation
- linting
- formatting
- static analysis

### Unit

- unit tests
- edge cases

### Integration

- component integration
- end-to-end tests

### Runtime

- crashes
- memory leaks
- resource usage
- timeouts

### Performance

- benchmarks
- profiling
- regression detection

### Security

- unsafe filesystem access
- command injection
- dependency risks
- sandbox escape attempts

---

# 16. Failure Recovery

Failures are expected.

Hermes must automatically recover from:

- compilation failures
- failing tests
- broken dependencies
- bad model outputs
- invalid patches
- benchmark regressions
- incomplete implementations
- agent loops
- resource exhaustion

Recovery strategy:

```text
Failure
  |
  v
Diagnose
  |
  v
Classify
  |
  +--> trivial --> retry
  |
  +--> implementation --> repair
  |
  +--> architecture --> reconsider
  |
  +--> research gap --> investigate
  |
  +--> repeated failure --> rollback
```

Repeated failures should trigger escalation to a stronger reasoning/research strategy rather than infinite retries.

---

# 17. Context Management

The project may become too large for a single model context.

Hermes must therefore maintain structured persistent context.

Important information should be stored in files rather than relying solely on conversation history.

Use:

```text
CONTEXT.md
AGENTS.md
SPEC.md
PLAN.md
docs/decisions/
research/
.hermes/state/
```

Agents should receive only the context required for their task whenever possible.

---

# 18. Model Router

Hermes should support multiple locally installed models.

The router should select a model according to task type.

Example:

```text
Research      -> reasoning/research model
Architecture  -> strongest reasoning model
Coding        -> coding model
Review        -> reasoning model
Simple tasks  -> smaller fast model
Evolution     -> coding/reasoning model
```

Model selection must be configurable.

Hermes must detect locally available models and validate that the requested model is actually installed.

No automatic fallback to cloud APIs is permitted.

---

# 19. Resource-Aware Execution

Hermes must be aware of available:

- CPU
- GPU
- RAM
- VRAM
- disk
- process limits

It should schedule agents accordingly.

For example:

```text
GPU-heavy inference
        +
CPU-heavy benchmark
        +
memory-heavy compilation
```

should not run concurrently if that would exceed available resources.

---

# 20. Experiment Management

Every experiment should be reproducible.

Record:

- experiment ID
- hypothesis
- configuration
- code version
- model used
- parameters
- environment
- result
- timestamp
- artifacts

Store results under:

```text
experiments/
.hermes/metrics/
```

Experiments must not silently overwrite previous results.

---

# 21. Reproducibility

Projects should be reproducible from a clean checkout.

Hermes should generate:

- setup instructions
- dependency definitions
- build commands
- test commands
- benchmark commands
- configuration examples

The final repository should be understandable without access to Hermes internals.

---

# 22. Git Integration

Hermes should initialize a Git repository automatically.

Commits should correspond to meaningful milestones.

Recommended commit strategy:

```text
feat: implement X
fix: repair Y
test: add Z
perf: improve benchmark A
docs: document B
refactor: simplify C
```

Hermes should be able to:

- create branches
- commit
- rollback
- compare candidates
- tag releases

---

# 23. GitHub Integration

GitHub upload should be optional and fully automated when enabled.

Hermes should support:

- repository creation
- initial push
- branch management
- tags/releases
- generated README
- issue synchronization where useful

GitHub credentials must never be hardcoded into generated projects.

---

# 24. Security Model

The factory must treat generated code as untrusted.

The sandbox should enforce:

- filesystem isolation
- network restrictions
- process isolation
- CPU limits
- memory limits
- disk limits
- execution timeouts

Network access should be disabled by default during implementation.

Research stages may use controlled network access if explicitly enabled by the factory configuration.

Generated code must not gain access to:

- SSH keys
- cloud credentials
- browser sessions
- environment secrets
- unrelated host files

---

# 25. Project Completion Criteria

Hermes must not declare a project complete merely because all tickets are marked completed.

Completion requires:

```text
Requirements satisfied
        AND
Build succeeds
        AND
Tests pass
        AND
Integration tests pass
        AND
No critical known defects
        AND
Performance requirements satisfied where applicable
        AND
Documentation exists
        AND
Repository is reproducible
```

The final verification should be executed from a clean environment.

---

# 26. Final Documentation

The final project must contain a useful README.

The README should include:

- project purpose
- features
- architecture
- installation
- requirements
- configuration
- usage
- examples
- testing
- benchmarks
- limitations
- license

README generation should happen **after** the implementation and final validation phases.

The final ticket in the project should therefore be:

```text
README.md
```

---

# 27. State Machine

Hermes itself should maintain an explicit project state.

Suggested states:

```text
INTAKE
RESEARCH
SPECIFICATION
PLANNING
IMPLEMENTATION
VALIDATION
OPTIMIZATION
DOCUMENTATION
RELEASE
COMPLETED
FAILED
```

State transitions must be persisted.

Hermes must be able to resume an interrupted project without losing progress.

---

# 28. Checkpointing

Create checkpoints after:

- research milestones
- architecture decisions
- successful tickets
- benchmark improvements
- major refactors
- release validation

A crash or machine restart must not destroy project progress.

---

# 29. Observability

Hermes should expose:

- current project state
- active ticket
- agent activity
- model being used
- resource usage
- test results
- benchmark results
- evolution generations
- failures
- retries

Logs should be machine-readable where practical.

---

# 30. CLI

Provide a CLI similar to:

```text
hermes new "project idea"
hermes run
hermes status
hermes pause
hermes resume
hermes logs
hermes inspect
hermes benchmark
hermes evolve
hermes test
hermes release
```

The normal workflow should be:

```text
hermes new "Build ..."
```

followed by autonomous execution.

---

# 31. Configuration

Configuration should allow:

- local model paths
- inference backend
- model selection
- GPU/CPU allocation
- RAM limits
- sandbox limits
- network policy
- maximum agent iterations
- research budget
- evolution budget
- benchmark budget
- GitHub integration
- logging level

Example conceptual configuration:

```yaml
models:
  coding:
    backend: llama.cpp
    model: /models/coding-model.gguf

  reasoning:
    backend: llama.cpp
    model: /models/reasoning-model.gguf

sandbox:
  runtime: docker
  network: restricted
  memory: 32g
  cpus: 16

research:
  enabled: true

evolution:
  enabled: true
  max_generations: 100

github:
  enabled: false
```

---

# 32. Dependency Philosophy

Hermes itself should minimize unnecessary dependencies.

Prefer:

- simple interfaces
- reproducible environments
- well-maintained open-source components
- local execution
- deterministic tooling

Do not introduce a dependency solely because an LLM recommends it.

Agents must justify major dependencies.

For Hermes itself the rule is stricter: §2.4 allows no third-party dependency beyond vLLM, and a useful tool's function is reimplemented instead.

---

# 33. Quality Gates

Before a ticket can enter `COMPLETED`:

```text
Implementation exists
        |
        v
Build succeeds
        |
        v
Tests pass
        |
        v
Acceptance criteria verified
        |
        v
Code reviewed
        |
        v
Ticket completed
```

For optimization tickets:

```text
Correctness
    +
Performance improvement
    +
No unacceptable regression
```

must all be satisfied.

---

# 34. Autonomous Decision Policy

When several technically valid options exist, Hermes should choose using:

1. correctness
2. security
3. simplicity
4. maintainability
5. reproducibility
6. performance
7. implementation cost

The factory should document significant decisions.

It should avoid unnecessary complexity.

---

# 35. Self-Improvement of Hermes

The factory itself should eventually be able to improve its own components.

Potential targets:

- planner
- ticket decomposition
- model routing
- prompt construction
- evaluation strategy
- candidate generation
- scheduling
- caching
- benchmark infrastructure

Self-modification must happen in a sandboxed branch with automated evaluation.

Only improvements that pass the Hermes test suite and demonstrate measurable benefit may be promoted.

---

# 36. Implementation Priority

Implement Hermes in the following order:

### Phase 1

Core project creation:

- CLI
- workspace generation
- state management
- configuration
- logging

### Phase 2

Agent execution:

- local model interface
- agent abstraction
- coding agent
- research agent
- reviewer

### Phase 3

Ticket system:

- Kanban
- dependency graph
- ticket lifecycle
- autonomous scheduler

### Phase 4

Sandbox:

- Docker integration
- resource limits
- command execution
- filesystem isolation

### Phase 5

Verification:

- test runner
- build runner
- evaluator
- regression detection

### Phase 6

Feynman research workflow:

- hypothesis management
- research tasks
- experiments
- knowledge synthesis

### Phase 7

OpenHands integration (superseded by §2.4: reimplement its function natively):

- optional OpenHands runner
- local-model configuration
- sandbox integration
- result ingestion

### Phase 8

AlphaEvolve-style optimization:

- baseline management
- candidate generation
- mutation
- evaluation
- selection
- generations
- rollback
- multi-objective optimization

### Phase 9

Git/GitHub:

- Git automation
- branches
- commits
- repository creation
- optional push

### Phase 10

Release system:

- clean-room validation
- documentation
- packaging
- final release

---

# 37. Definition of Done for Hermes

Hermes is considered production-ready when it can autonomously execute a non-trivial project from a single idea prompt.

A complete demonstration should show:

```text
Single user prompt
        |
        v
Automatic research
        |
        v
Automatic specification
        |
        v
Automatic planning
        |
        v
Automatic ticket generation
        |
        v
Automatic implementation
        |
        v
Automatic testing
        |
        v
Automatic debugging
        |
        v
Automatic research iteration
        |
        v
Automatic AlphaEvolve-style optimization
        |
        v
Automatic final validation
        |
        v
Automatic documentation
        |
        v
Git repository
        |
        v
Optional GitHub repository
```

No human decision should be required between these stages.

---

# 38. Engineering Principle

The central design principle is:

> Hermes should behave less like a chatbot and more like an autonomous software organization operating inside a controlled laboratory.

The system should continuously:

```text
Observe
Research
Plan
Act
Measure
Evaluate
Learn
Improve
Repeat
```

until the project reaches objectively verifiable completion.

---

# 39. Scheduled Upkeep and Community Improvement

When the factory goes live, its operator chooses how often it looks after itself: weekly, biweekly, monthly, or another interval, or not at all. This is asked once, at setup, never during autonomous work (§2.2).

On that schedule the factory:

1. crawls the web for add-ons — tools whose function would measurably benefit the factory — and records the function to rebuild, never the code (§2.4);
2. starts a factory optimisation run on itself: self-modifications, including rebuilt add-on functions, judged by the Hermes test suite and by a measured before/after evaluation on the operator's hardware (§35).

When a run on a user's clone finds a general improvement — one that holds on the whole evaluation suite — the factory automatically opens a pull request upstream, and that pull request adds the user to the list of improvers, because they contributed their hardware to the project.

Contribution is opt-in at setup, uses the operator's own credentials, and never sends the operator's unpushed work. Upstream maintainers review every such pull request; nothing merges itself.
