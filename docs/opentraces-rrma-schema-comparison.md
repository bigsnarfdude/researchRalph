# OpenTraces x RRMA: Schema Comparison & Extension Proposal

Working document for proposing multi-agent extensions to opentraces v0.1.0.

---

## A. Field-by-Field Mapping

### TraceRecord (top-level)

| OpenTraces field | Type | RRMA equivalent | Notes |
|---|---|---|---|
| `schema_version` | `str` | None | RRMA has no schema versioning; logs are raw Claude Code JSONL |
| `trace_id` | `str` | None (derive from `logs/agent{N}.jsonl` filename + timestamp) | Gap: RRMA has no unique trace ID per session |
| `session_id` | `str` | Screen session name: `rrma-worker{N}` or `rrma-meta` | 1:1 map possible |
| `content_hash` | `str` | None | Gap: no dedup mechanism |
| `timestamp_start` | `str` | First line timestamp in `logs/agent{N}.jsonl` | Extractable |
| `timestamp_end` | `str` | Last line timestamp / screen session end | Extractable |
| `task` | `Task` | See below | Partial |
| `agent` | `Agent` | See below | Partial |
| `environment` | `Environment` | See below | Partial |
| `system_prompts` | `dict[str,str]` | The `-p` prompt string in `launch-agents.sh` (line 65) | One prompt per agent; gardener prompts differ per decision type |
| `tool_definitions` | `list[dict]` | Claude Code's built-in tools (Read, Edit, Bash, etc.) | Implicit; not recorded in RRMA logs |
| `steps` | `list[Step]` | JSONL lines in `logs/agent{N}.jsonl` | See Step mapping below |
| `outcome` | `Outcome` | See below | Partial |
| `dependencies` | `list[str]` | Domain files: `config.yaml`, `sae.py`, `engine.py`, etc. | Not tracked explicitly |
| `metrics` | `Metrics` | See below | Partial |
| `security` | `SecurityMetadata` | None | Gap: no security scanning |
| `attribution` | `Attribution` | None | Gap: agents edit shared files; attribution is ambiguous |
| `metadata` | `dict` | `config.yaml` (domain params), `AGENT_ID` env var | Extensible |

### Task

| OpenTraces field | Type | RRMA equivalent |
|---|---|---|
| `task.description` | `str` | `program.md` content (the agent instructions) |
| `task.source` | `str` | `"rrma_v4_scaffold"` (static) |
| `task.repository` | `str` | Domain directory path, e.g. `domains/sae-bench` |
| `task.base_commit` | `str` | Git HEAD at launch time (not currently captured) |

### Agent

| OpenTraces field | Type | RRMA equivalent |
|---|---|---|
| `agent.name` | `str` | `"claude-code"` (all agents use same CLI) |
| `agent.version` | `str` | Claude CLI version (not captured) |
| `agent.model` | `str` | Model used by Claude CLI (not captured; set by env/config) |

**Gap**: RRMA needs `agent_id` (e.g. `agent0`..`agent7`) which is the swarm-local identity. OpenTraces has no concept of multiple agents within a coordinated run.

### Environment

| OpenTraces field | Type | RRMA equivalent |
|---|---|---|
| `environment.os` | `str` | Host OS (available but not captured) |
| `environment.shell` | `str` | `bash` (always) |
| `environment.vcs.type` | `str` | `"git"` |
| `environment.vcs.base_commit` | `str` | Git HEAD at generation start |
| `environment.vcs.branch` | `str` | Current branch |
| `environment.vcs.diff` | `str` | Not applicable (agents edit domain files, not a git diff) |
| `environment.language_ecosystem` | `list[str]` | From domain: `["python"]` typically |

### Step

| OpenTraces field | Type | RRMA equivalent |
|---|---|---|
| `step.step_index` | `int` | Line number in agent JSONL |
| `step.role` | `"system"\|"user"\|"agent"` | JSONL `type` field: `user` or `assistant`; thinking blocks exist too |
| `step.content` | `str` | JSONL `message.content` text blocks |
| `step.reasoning_content` | `str` | JSONL thinking blocks (Claude extended thinking) |
| `step.model` | `str` | Not per-step in RRMA (same model throughout) |
| `step.system_prompt_hash` | `str` | Derivable from the `-p` flag content |
| `step.agent_role` | `str` | **Overloaded in RRMA**: `"worker"`, `"meta"`, `"gardener"`, `"conductor"` |
| `step.parent_step` | `int` | None within a single agent; see cross-agent refs below |
| `step.call_type` | `str` | `"main"` for workers, `"subagent"` for conductor-spawned agents |
| `step.subagent_trajectory_ref` | `str` | None (agents don't spawn sub-agents; the gardener spawns agents) |
| `step.tools_available` | `list[str]` | Claude Code tools (implicit, not logged) |
| `step.tool_calls` | `list[ToolCall]` | JSONL `tool_use` blocks |
| `step.observations` | `list[Observation]` | JSONL `tool_result` blocks |
| `step.snippets` | `list[Snippet]` | Not extracted |
| `step.token_usage` | `TokenUsage` | Available in Claude Code JSONL stream-json output |
| `step.timestamp` | `str` | Available in JSONL |

### Outcome

| OpenTraces field | Type | RRMA equivalent |
|---|---|---|
| `outcome.success` | `bool` | Derived: did the agent's experiments improve `results.tsv`? |
| `outcome.signal_source` | `str` | `"harness"` (run.sh evaluator) |
| `outcome.signal_confidence` | `str` | `"derived"` (deterministic from harness output) |
| `outcome.description` | `str` | Best score achieved, experiment count |
| `outcome.patch` | `str` | Not applicable (agents produce configs, not patches) |
| `outcome.committed` | `bool` | N/A |
| `outcome.commit_sha` | `str` | N/A |

### Metrics

| OpenTraces field | Type | RRMA equivalent |
|---|---|---|
| `metrics.total_steps` | `int` | Line count of agent JSONL |
| `metrics.total_input_tokens` | `int` | Sum from JSONL token_usage fields |
| `metrics.total_output_tokens` | `int` | Sum from JSONL token_usage fields |
| `metrics.total_duration_s` | `float` | `timestamp_end - timestamp_start` |
| `metrics.cache_hit_rate` | `float` | Derivable from JSONL |
| `metrics.estimated_cost_usd` | `float` | Not tracked |

---

## B. RRMA Concepts That Don't Fit in OpenTraces

These are first-class concepts in RRMA with no OpenTraces equivalent:

### 1. Swarm Identity

Multiple agents run concurrently as a coordinated swarm. Each has:
- `agent_id`: local identity (`agent0`..`agent7`)
- `swarm_run_id`: the overall outer-loop invocation
- `generation`: which generation of the outer loop (1..N)
- `domain`: which benchmark/task they operate on

OpenTraces assumes **one agent = one trace**. RRMA needs **one swarm run = many traces + coordination artifacts**.

### 2. Blackboard (Shared Mutable State)

- Append-only markdown file read/written by all agents
- Contains: `CLAIM`, `RESPONSE`, `REQUEST` entries
- Compressed periodically by the meta-agent
- Versioned implicitly by generation (backed up as `blackboard.md.gen{N}`)
- Carries structured sub-files: `DESIRES.md`, `LEARNINGS.md`, `MISTAKES.md`

No OpenTraces equivalent. This is a shared artifact that influences all agents but belongs to none.

### 3. Meta-Blackboard (Compressed Memory)

- Written by the meta-agent every N minutes
- Distills blackboard + results into structured sections: What Works, Dead Ends, Blind Spots, Surprises, Devil's Advocate
- Persists across generations (cross-generation memory)
- Agents optionally read it; it's advisory, not directive

### 4. Gardener Decisions

The outer loop (gardener) makes periodic assessments:
- **Process Quality score** (0-30): papers cited, explanations, ablations, simplifications, etc.
- **Decision**: `CONTINUE | TOO_EARLY | NUDGE | STOP_HACKING | STOP_DONE | REDESIGN`
- **Actions taken**: rewrite `program.md`, append nudge to blackboard, stop agents
- **Inputs read**: blackboard, results.tsv, meta-blackboard, DESIRES/LEARNINGS/MISTAKES

This is a supervisory control loop, not an agent session. It has no steps/tools -- it reads artifacts and makes structural decisions.

### 5. Generation Boundaries

A "generation" is:
- One invocation of `launch-agents.sh` through `stop-agents.sh`
- Bounded by calibrate (start) and diagnose (end)
- Has pre/post metrics: experiment count, best score
- May trigger scaffold rewrites between generations
- Agents within a generation share state; across generations, only meta-blackboard persists

### 6. Cross-Agent Influence

Causal chains across agents:
- Agent2 writes finding to blackboard -> Agent5 reads it -> Agent5's next experiment uses it
- Agent3 posts `REQUEST` -> conductor spawns ephemeral agent to handle it
- Gardener appends `NUDGE` to blackboard -> all agents see it on next read
- Meta-agent compresses blackboard -> agents read compressed version

These are not parent-child relationships (OpenTraces' `parent_step`). They are **broadcast influence via shared artifacts**.

### 7. Scaffold Evolution

`program.md` (agent instructions) can be rewritten by the gardener between generations. The same agent role receives different instructions across generations. This is a meta-level change -- the "task description" itself evolves.

### 8. Domain Harness

The evaluation harness (`run.sh`) and its output (`results.tsv`) are first-class:
- `results.tsv` columns: `commit, score, memory_gb, status, description, agent, design`
- Direction: `higher_is_better` or `lower_is_better`
- Solved/unsolved blacklists: `solved.txt`, `unsolved.txt`

OpenTraces' `Outcome` captures one outcome per session. RRMA agents produce **many scored experiments per session**.

### 9. Agent Self-Telemetry

Per-agent files that agents maintain:
- `DESIRES.md`: tools or context they wish they had
- `MISTAKES.md`: tactics that failed and why
- `LEARNINGS.md`: discoveries about the environment

These feed into the gardener's PQ scoring and NUDGE/REDESIGN decisions.

---

## C. Minimal Schema Extensions

The goal: support multi-agent research swarms with the **smallest** set of additions to OpenTraces v0.1.0, without breaking single-agent compatibility.

### Extension 1: SwarmRecord (new top-level model)

```python
class SwarmRecord(BaseModel):
    """Wraps multiple TraceRecords from a coordinated multi-agent run."""

    schema_version: str = SCHEMA_VERSION
    swarm_id: str                          # UUID for the entire run
    swarm_type: str                        # "rrma_v4", "metagpt", "autogen", etc.
    domain: str                            # benchmark/task identifier
    generation: int                        # which outer-loop generation
    generation_total: int                  # max generations configured

    # Timing
    timestamp_start: str
    timestamp_end: str | None = None

    # Agent traces (each is a standard TraceRecord)
    traces: list[str] = Field(
        default_factory=list,
        description="trace_ids of member TraceRecords (stored separately in JSONL)"
    )

    # Coordination role map
    roles: dict[str, str] = Field(
        default_factory=dict,
        description="trace_id -> role: worker|meta|gardener|conductor"
    )

    # Shared artifacts (see Extension 2)
    artifacts: list[SharedArtifact] = Field(default_factory=list)

    # Supervisory decisions (see Extension 3)
    decisions: list[SupervisoryDecision] = Field(default_factory=list)

    # Aggregate metrics
    total_experiments: int = 0
    best_score: float | None = None
    score_direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"
    process_quality: int | None = Field(None, ge=0, le=30)

    # Scaffold versioning
    scaffold_versions: list[ScaffoldVersion] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)
```

### Extension 2: SharedArtifact (blackboard, meta-blackboard, results)

```python
class ArtifactSnapshot(BaseModel):
    """A point-in-time snapshot of a shared artifact."""

    timestamp: str
    content_hash: str
    line_count: int
    trigger: str                           # "meta_compression", "generation_end",
                                           # "nudge_append", "gardener_reset"
    content: str | None = None             # full content (optional; may be large)
    delta: str | None = None               # diff from previous snapshot (preferred)

class SharedArtifact(BaseModel):
    """A mutable artifact shared across agents in a swarm."""

    artifact_id: str
    artifact_type: Literal[
        "blackboard",          # main shared state
        "meta_blackboard",     # compressed memory
        "results",             # scored experiment log
        "desires",             # agent tool/context requests
        "learnings",           # discovered knowledge
        "mistakes",            # failure patterns
        "program",             # agent instructions (scaffold)
        "calibration",         # pre-run literature search
    ]
    path: str                              # relative path within domain
    snapshots: list[ArtifactSnapshot] = Field(default_factory=list)
```

### Extension 3: SupervisoryDecision (gardener actions)

```python
class SupervisoryDecision(BaseModel):
    """A control decision made by an outer-loop supervisor (gardener)."""

    timestamp: str
    decision: Literal[
        "CONTINUE", "TOO_EARLY", "NUDGE",
        "STOP_HACKING", "STOP_DONE", "REDESIGN"
    ]
    process_quality: int = Field(ge=0, le=30)
    generation: int

    # Inputs that informed the decision
    total_experiments: int
    best_score: float | None = None
    stagnation_depth: int = 0              # experiments since best
    flat: bool = False                     # rolling window stagnation
    blind_spots: int = 0
    scaffold_desires: int = 0              # agent requests for tooling

    # PQ breakdown
    pq_signals: dict[str, int] = Field(
        default_factory=dict,
        description="papers, explanations, ablations, simplifications, etc."
    )

    # Action taken (if any)
    action: str | None = None              # "rewrote program.md", "appended nudge", etc.
    action_content: str | None = None      # the nudge text or new program.md hash
```

### Extension 4: Step-level additions (minimal, backward-compatible)

Add to existing `Step` model:

```python
class Step(BaseModel):
    # ... existing fields ...

    # Multi-agent extensions (all optional, backward-compatible)
    swarm_agent_id: str | None = Field(
        None,
        description="Agent's local identity within the swarm: agent0, agent1, meta, gardener"
    )
    artifact_reads: list[str] = Field(
        default_factory=list,
        description="artifact_ids read during this step (e.g. blackboard, meta_blackboard)"
    )
    artifact_writes: list[str] = Field(
        default_factory=list,
        description="artifact_ids written during this step"
    )
    experiment_result: ExperimentResult | None = Field(
        None,
        description="Scored experiment produced by this step (multi-experiment sessions)"
    )
```

### Extension 5: ExperimentResult (multi-outcome per session)

```python
class ExperimentResult(BaseModel):
    """A single scored experiment within a session.

    RRMA agents produce many experiments per session. Each is a
    config change + harness evaluation, not a code patch.
    """

    experiment_id: str                     # from results.tsv commit column
    score: float
    status: Literal["keep", "discard", "crash", "retest"]
    description: str | None = None
    design: str | None = None              # experiment design category
    agent_id: str | None = None            # which swarm agent ran it
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    memory_gb: float | None = None
```

### Extension 6: CrossAgentInfluence (causal provenance)

```python
class CrossAgentInfluence(BaseModel):
    """Records causal influence between agents via shared artifacts.

    This is NOT a parent-child hierarchy (which OpenTraces already handles
    via parent_step). This captures broadcast influence: agent A writes to
    blackboard, agent B reads it and changes behavior.
    """

    source_trace_id: str                   # trace that wrote the artifact
    source_step_index: int                 # step that performed the write
    target_trace_id: str                   # trace that read the artifact
    target_step_index: int                 # step that performed the read
    artifact_id: str                       # which shared artifact
    influence_type: Literal[
        "finding",             # agent read another's experimental finding
        "request",             # agent fulfilled another's REQUEST
        "nudge",               # gardener nudge read by agent
        "compression",         # meta-agent compression read by agent
        "scaffold_rewrite",    # gardener rewrote program.md, agent got new instructions
    ]
    confidence: Literal["high", "medium", "low"] = "medium"
```

### Summary: What Gets Added Where

| Addition | Scope | Breaks existing? |
|---|---|---|
| `SwarmRecord` | New top-level model alongside `TraceRecord` | No (additive) |
| `SharedArtifact` + `ArtifactSnapshot` | New models referenced by `SwarmRecord` | No |
| `SupervisoryDecision` | New model referenced by `SwarmRecord` | No |
| `ExperimentResult` | New model referenced by `Step` | No |
| `CrossAgentInfluence` | New model referenced by `SwarmRecord` | No |
| `Step.swarm_agent_id` | Optional field on existing model | No (nullable) |
| `Step.artifact_reads/writes` | Optional fields on existing model | No (default empty) |
| `Step.experiment_result` | Optional field on existing model | No (nullable) |

Total: 6 new models, 4 new optional fields on `Step`. Zero breaking changes.

---

## D. Draft PR Description

### Title

`feat(schema): multi-agent swarm extensions for coordinated research systems`

### Body

```
## Summary

Proposes extensions to opentraces-schema v0.1.0 to support multi-agent
swarms where N agents coordinate through shared artifacts rather than
parent-child hierarchies.

Motivated by RRMA (Recursive Research Multi-Agent), a framework where
4-8 agents run concurrently on a research benchmark, share findings via
an append-only blackboard, and are supervised by a gardener outer loop
that can stop/nudge/redesign the scaffold between generations.

## Problem

The current schema models one agent = one trace, with multi-agent support
limited to parent_step hierarchies (main agent spawns sub-agents). This
misses a common multi-agent pattern: **peer coordination through shared
mutable state**, where:

- Multiple agents write to a shared blackboard (not spawned by each other)
- A supervisory loop makes structural decisions (not tool calls)
- Agents produce many scored experiments per session (not one outcome)
- The task description itself evolves across generations
- Causal influence flows through artifacts, not call stacks

## Proposed Changes

**New models (all additive, zero breaking changes):**

1. **SwarmRecord** — wraps multiple TraceRecords from one coordinated run.
   References traces by ID, adds generation boundaries, aggregate metrics,
   and score direction.

2. **SharedArtifact / ArtifactSnapshot** — versioned snapshots of shared
   mutable state (blackboard, results, agent instructions). Captures the
   coordination medium that single-agent tracing ignores.

3. **SupervisoryDecision** — structured gardener/orchestrator decisions
   with process quality scores and PQ signal breakdowns. Not a Step
   (no tool calls), not an Outcome (influences future generations).

4. **ExperimentResult** — a single scored experiment within a session.
   Supports agents that run many evaluate-and-score cycles per trace,
   not just one patch.

5. **CrossAgentInfluence** — causal provenance linking a write in one
   trace to a read in another, via a shared artifact. Captures broadcast
   influence that parent_step cannot represent.

**Step-level additions (all optional, backward-compatible):**

- `swarm_agent_id`: local identity within the swarm
- `artifact_reads` / `artifact_writes`: which shared artifacts this step touched
- `experiment_result`: inline scored experiment

## Design Principles

- **Additive only**: existing single-agent traces are unchanged
- **SwarmRecord references TraceRecords**: no duplication; a swarm is a
  collection of standard traces plus coordination metadata
- **Artifacts are the coordination primitive**: rather than inventing a
  new message-passing schema, we model the shared state that agents
  actually read and write
- **Supervisory decisions are not steps**: they operate at a different
  timescale and have different inputs (aggregate metrics, not tool results)

## Test Plan

- [ ] Validate that existing TraceRecord serialization is unchanged
- [ ] Round-trip SwarmRecord with 4 traces through to_jsonl_line()
- [ ] Verify CrossAgentInfluence references resolve to valid trace/step pairs
- [ ] Export SwarmRecord to ATIF (falls back to per-trace export, swarm
      metadata in ATIF extras)
- [ ] Parse actual RRMA v4 run (8 agents, 3 generations, 186 experiments)
      into SwarmRecord + TraceRecords
```

---

## Appendix: RRMA Data Flow Diagram

```
outer-loop.sh (gardener)
    │
    ├── calibrate.sh ──────────────► calibration.md (SharedArtifact)
    │
    ├── launch-agents.sh
    │   ├── agent0 (TraceRecord) ──┐
    │   ├── agent1 (TraceRecord) ──┤
    │   ├── agent2 (TraceRecord) ──┼──► blackboard.md (SharedArtifact, append-only)
    │   ├── agent3 (TraceRecord) ──┤    DESIRES.md, LEARNINGS.md, MISTAKES.md
    │   └── meta   (TraceRecord) ──┘──► meta-blackboard.md (SharedArtifact, replaced)
    │
    ├── diagnose.sh ───────────────► SupervisoryDecision
    │   reads: blackboard, results, meta-blackboard, DESIRES, LEARNINGS, MISTAKES
    │   outputs: PQ score (0-30) + decision
    │
    ├── [NUDGE] ───────────────────► blackboard.md append (CrossAgentInfluence)
    ├── [REDESIGN] ────────────────► program.md rewrite (ScaffoldVersion)
    ├── [STOP_HACKING] ───────────► program.md rewrite + blackboard reset
    └── [STOP_DONE] ──────────────► final meta-blackboard + taste.md lesson

    results.tsv: experiment_id | score | memory_gb | status | description | agent | design
    (one row per ExperimentResult, many per TraceRecord)
```

---

## Open Questions

1. **Artifact content vs. reference**: Should `SharedArtifact.snapshots` store full content inline, or reference external files? For blackboards that reach 500+ lines, inline is expensive. Proposal: `content` is optional, `content_hash` + `delta` preferred.

2. **CrossAgentInfluence granularity**: Should we require step-level precision (which step read which artifact), or is trace-level sufficient? Step-level is more useful for causal analysis but harder to extract (requires parsing tool_use for file reads).

3. **Gardener as TraceRecord?**: The gardener invokes Claude CLI for nudges and redesigns. Should those be their own TraceRecords with `agent_role: "gardener"`, or are SupervisoryDecisions sufficient? Proposal: both -- TraceRecord for the Claude calls, SupervisoryDecision for the structured output.

4. **Generation boundaries in JSONL**: Should SwarmRecords be interleaved with TraceRecords in the same JSONL file, or stored in a separate file? Proposal: separate file (`swarm.jsonl` alongside `traces.jsonl`).

5. **results.tsv schema standardization**: Should ExperimentResult subsume the TSV format, or should the TSV remain the source of truth with ExperimentResult as a parsed view? Proposal: ExperimentResult is the schema; TSV is a serialization format for backward compatibility.
