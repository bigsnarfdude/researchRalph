# RRMA (Research Ralph Multi-Agent) — Full Architecture

**Version:** v4.7 | **Author:** Vincent Oh | **Last Updated:** April 2026

---

## Table of Contents

1. [Overview](#overview)
2. [System Diagram](#system-diagram)
3. [The Gardener (outer-loop.sh)](#the-gardener)
4. [Worker Agents](#worker-agents)
5. [The Meta-Agent](#the-meta-agent)
6. [TrustLoop (Behavioral IDS)](#trustloop)
7. [Process Quality & Decision Matrix](#process-quality--decision-matrix)
8. [Context Optimization (v4.6)](#context-optimization)
9. [Memory System (v4.7)](#memory-system)
10. [Domain File Layout](#domain-file-layout)
11. [End-to-End Data Flow](#end-to-end-data-flow)
12. [MCP Servers](#mcp-servers)
13. [Gardener Principles (taste.md)](#gardener-principles)
14. [Hardware](#hardware)
15. [Proven Results](#proven-results)

---

## Overview

RRMA is a multi-agent research framework where Claude Code agents run experiments autonomously on a shared blackboard. A "gardener" outer agent monitors process quality, detects hacking/stagnation, and redesigns the scaffold between generations. TrustLoop provides forensic scoring, anomaly detection, and insight generation.

**Key idea:** Agents explore a research problem in parallel. They share findings via append-only files. A gardener watches from above and intervenes only when the process breaks down — never micromanaging, only course-correcting.

```
Human
  │
  ▼
outer-loop.sh (Gardener) ──── taste.md (learned principles)
  │
  ├── calibrate.sh ──────────► calibration.md (literature search, gen 1 only)
  │
  ├── launch-agents.sh
  │     ├── Worker Agent 0 ──► workspace/agent0/ (isolated config)
  │     ├── Worker Agent 1 ──► workspace/agent1/
  │     ├── Worker Agent N ──► workspace/agentN/
  │     └── Meta-Agent ──────► meta-blackboard.md (reflections)
  │
  ├── Monitor Loop (every N min)
  │     ├── refresh_context.py ──► stoplight.md + recent_experiments.md
  │     └── diagnose.py ─────────► decision (CONTINUE/NUDGE/REDESIGN/STOP)
  │           └── trustloop_scorer.py (classification + anomaly detection)
  │
  └── Decision Handler
        ├── NUDGE ──────► inject observation into blackboard + program.md
        ├── STOP_HACKING ► rewrite program.md (force genuine research)
        ├── REDESIGN ───► diagnose scaffold block, minimal fix
        └── STOP_DONE ──► final meta-blackboard + lesson → taste.md
```

---

## System Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           outer-loop.sh (THE GARDENER)                   │
│                                                                          │
│  Reads: taste.md (principles from prior runs)                            │
│  Writes: outer-loop.log, meta-blackboard.md, taste.md (appends lessons) │
│  Backs up: blackboard.md.genN each generation                            │
│                                                                          │
│  ┌─── Gen 1 Only ────────────────────────────────────┐                   │
│  │  calibrate.sh                                      │                   │
│  │  └─► Claude + WebSearch → calibration.md           │                   │
│  │      (SOTA, papers, techniques, baselines)         │                   │
│  └────────────────────────────────────────────────────┘                   │
│                                                                          │
│  ┌─── Per Generation ────────────────────────────────────────────────┐   │
│  │                                                                    │   │
│  │  STEP 1: launch-agents.sh                                         │   │
│  │  ├─ refresh_context.py → stoplight.md + recent_experiments.md     │   │
│  │  ├─ memory_system.py seed → domain/memory/                        │   │
│  │  ├─ memory_system.py recall → memory context per agent            │   │
│  │  ├─ Create workspace/agent0/, agent1/ ... (isolated train.py)     │   │
│  │  ├─ Spawn N workers in screen sessions (15s apart)                │   │
│  │  └─ Spawn meta-agent (meta-loop.sh in screen)                     │   │
│  │                                                                    │   │
│  │  STEP 2: Monitor Loop (every N minutes)                            │   │
│  │  ├─ Check: are workers still alive?                                │   │
│  │  ├─ refresh_context.py → update stoplight + recent_experiments    │   │
│  │  ├─ diagnose.py → trustloop_scorer → PQ score → decision         │   │
│  │  │                                                                 │   │
│  │  │  Decision routing:                                              │   │
│  │  │  ├─ CONTINUE     → keep monitoring                             │   │
│  │  │  ├─ TOO_EARLY    → keep monitoring (< 8 experiments)           │   │
│  │  │  ├─ NUDGE        → inject observation + constraints            │   │
│  │  │  │                  (max 3, then escalate to REDESIGN)         │   │
│  │  │  ├─ STOP_HACKING → rewrite program.md for genuine research    │   │
│  │  │  ├─ REDESIGN     → diagnose scaffold block → minimal fix      │   │
│  │  │  └─ STOP_DONE    → check for unexplored dirs                  │   │
│  │  │                     if found → downgrade to NUDGE              │   │
│  │  │                     else → final meta-blackboard + taste.md    │   │
│  │  │                                                                 │   │
│  │  STEP 3: stop-agents.sh (kill all screen sessions)                 │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Loop back for next generation with updated program.md                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## The Gardener

**File:** `v4/outer-loop.sh`

The gardener is the top-level orchestrator. It manages multi-generation research runs, making high-level decisions about when to intervene, when to stop, and when to redesign the scaffold.

**Inputs:**
- Domain path (required)
- Max generations, number of agents, max turns per agent, monitor interval

**Per-generation workflow:**
1. (Gen 1 only) Run `calibrate.sh` for literature search → `calibration.md`
2. Call `launch-agents.sh` to start workers + meta-agent in screen sessions
3. Monitor loop: every N minutes check workers, refresh context, run diagnosis
4. Act on diagnosis decision (NUDGE / REDESIGN / STOP_HACKING / STOP_DONE)
5. Kill workers, back up blackboard, start next generation if needed

**Key principle:** The gardener never tells agents *what* to try. It adjusts the *scaffold* (program.md constraints, closed brackets, regime) so agents self-correct.

---

## Worker Agents

Each worker is an independent Claude Code instance running in a screen session.

```
┌─────────────────────────────────────────────────────────────────┐
│                  WORKER AGENT (1 of N)                           │
│                  Screen: rrma-workerN                            │
│                  Log: logs/agentN.jsonl                          │
│                                                                  │
│  READS (on startup, in order):                                   │
│  1. program_static.md  ← immutable rules (read once)            │
│  2. program.md         ← dynamic guidance (gardener rewrites)   │
│  3. stoplight.md       ← 30-line compressed run state           │
│  4. recent_experiments.md ← last 5 experiments, structured      │
│  5. best/train.py      ← current best configuration             │
│  6. meta-blackboard.md ← meta-agent reflections (if exists)     │
│  7. calibration.md     ← literature baseline (if exists)        │
│  8. [memory context]   ← from memory_system.py recall           │
│                                                                  │
│  EXPERIMENT LOOP (repeats until max_turns):                      │
│  1. Read stoplight → identify gap or hypothesis                  │
│  2. Edit workspace/agentN/train.py (own copy only)              │
│  3. Run: bash run.sh <name> "<description>" <design_type>       │
│     └─ run.sh resolves workspace via $CLAUDE_AGENT_ID           │
│     └─ GPU serialized via flock                                  │
│  4. Append result to results.tsv                                 │
│  5. Append findings to blackboard.md                             │
│  6. Update MISTAKES.md / DESIRES.md / LEARNINGS.md              │
│  7. Repeat                                                       │
│                                                                  │
│  results.tsv format:                                             │
│  id  score  keep/discard  "description"  agent  design  time    │
└─────────────────────────────────────────────────────────────────┘
```

**v4.7 workspace isolation:** Each agent gets a private copy of `best/train.py` in `workspace/agentN/`. This eliminates the #1 race condition from earlier versions where agents overwrote each other's configs.

---

## The Meta-Agent

**File:** `v4/meta-loop.sh` | **Screen:** `rrma-meta`

The meta-agent observes and reflects but **never directs agents**. It provides shared memory via `meta-blackboard.md`.

```
Every N minutes:
├─ refresh_context.py → update stoplight + recent_experiments
├─ Read: stoplight.md, recent_experiments.md, best/config.yaml
├─ Read: previous meta-blackboard.md (if exists)
└─ Claude (3 turns) → generate new meta-blackboard.md
```

**meta-blackboard.md contains (~120 lines max):**
- Current best + config
- What works (ranked by impact)
- Dead ends (grouped by category)
- Patterns noticed (process-level, not just results)
- Blind spots (never-tried approaches)
- Stepping stones (non-winning but promising experiments)
- Surprises (expected vs actual)
- Devil's advocate (strongest case the best score is misleading)
- Self-reflection (compare to prior cycle)

Written atomically (`.tmp` + `mv`) to prevent partial reads.

---

## TrustLoop

**File:** `tools/trustloop_scorer.py`

TrustLoop is the central nervous system — it analyzes every experiment and feeds all diagnosis, nudge, and insight generation.

```
INPUT: results.tsv, blackboard.md, MISTAKES.md, DESIRES.md, LEARNINGS.md

OUTPUT: DomainReport containing:

1. Experiment Classification
   BREAKTHROUGH │ INCREMENTAL │ PLATEAU │ REGRESSION │ CRASH

2. Novelty Score (0-1 per experiment)
   70% description similarity + 30% design label match

3. Agent Efficiency
   Success rate, waste ratio, best contribution per agent

4. Redundancy Detection
   Near-duplicate configs flagged

5. Anomaly Detection
   - Crash streaks (3+ consecutive)
   - Deep stagnation (30+ experiments without breakthrough)
   - Score jumps (unlikely large changes)
   - Resource waste (high redundancy)

6. Workflow Checks (14 checks)
   Agent diversity, blackboard usage, format validation

7. Insight Extraction
   Winning strategies, dead ends, recurring mistakes, unaddressed desires

8. Telemetry Parsing
   Structured MISTAKES, DESIRES, LEARNINGS content

9. Action Items
   Owner: hitl | gardener
   Layer: harness | program.md | agent | scaffold
```

**Consumed by:**
- `diagnose.py` → PQ score + decision logic
- `refresh_context.py` → stoplight + recent_experiments generation
- `trustloop_mcp.py` → Claude Code inspection tools

---

## Process Quality & Decision Matrix

**File:** `v4/diagnose.py`

### Process Quality (PQ) Score: 0–30

| Indicator | Points |
|-----------|--------|
| Papers cited | +3 (>3 papers: +3 more) |
| Explanatory reasoning | +3 (>10 explanations: +3 more) |
| Ablations | +3 (>3 ablations: +3 more) |
| Simplifications | +3 |
| Design diversity | +3 (>5 unique designs) |
| Blackboard usage | +3 (>100 lines) |
| Desires written | +3 |
| Learnings written | +3 (>5 learnings) |

### Decision Matrix

| Condition | Decision | Action |
|-----------|----------|--------|
| < 8 experiments | `TOO_EARLY` | Keep monitoring |
| PQ < 10, > 15 experiments | `STOP_HACKING` | Rewrite program.md (force papers, ablations, explanations) |
| Crash streak or scaffold desires | `NUDGE` | Inject observation + constraints |
| Stagnation without flatness | `NUDGE` | Inject observation + constraints |
| 3+ nudges without progress | escalate | → `REDESIGN` |
| Flat + PQ ≥ 10 + blind spots | `REDESIGN` | Diagnose scaffold block → minimal fix to program.md |
| Flat + PQ ≥ 10 + no blind spots | `STOP_DONE` | Re-evaluate; if unexplored dirs found → NUDGE; else finalize |
| Otherwise | `CONTINUE` | Keep monitoring |

**Output:** decision string + `.nudge_data.json` (gardener_fixes, dead_ends, tool_issues, dominant_axis, stagnation)

---

## Context Optimization

**Introduced in v4.6.** Agents no longer read 600+ lines of raw blackboard. Instead:

| Old (v4.5) | New (v4.6) | Lines |
|------------|------------|-------|
| `program.md` (monolithic 261 lines) | `program_static.md` (read once) + `program.md` (dynamic) | 98 + 95 |
| `blackboard.md` (627+ lines, re-read every cycle) | `stoplight.md` (30 lines, auto-refreshed) | 43 |
| grep results.tsv (growing) | `recent_experiments.md` (last 5, structured) | ~30 |

**File:** `tools/refresh_context.py`

Generates two compact files called by both the outer-loop and meta-loop:

- **`stoplight.md` (~30 lines):** Best score, experiment count, stagnation depth, what works (top 5), dead ends (top 8), recurring problems, gaps, agent summary, alerts
- **`recent_experiments.md`:** Structured per-experiment records with ID, score, outcome class, agent, design, verdict, delta from best

---

## Memory System

**File:** `tools/memory_system.py` | **Introduced in v4.7**

Filesystem-native memory retrieval — no vector DB, no embeddings.

### Three Subsystems

1. **Scanner:** Parse frontmatter + mtime from `.md` files in `domain/memory/` → manifest
2. **Retriever:** Haiku side-query picks top-5 relevant files per query; keyword fallback when LLM unavailable
3. **Staleness Checker:** Age-based warning levels

### Staleness Levels

| Age | Level | Behavior |
|-----|-------|----------|
| ≤ 1 day | Fresh | Used as-is |
| 1–7 days | Recent | Used as-is |
| 7–30 days | Aging | Wrapped with verification warning |
| > 30 days | Stale | Wrapped with verification warning |

### Commands

```bash
memory_system.py seed <domain>           # Initialize domain/memory/
memory_system.py scan <memory_dir>       # Print manifest
memory_system.py retrieve <dir> <query>  # LLM-select + load
memory_system.py recall <dir> <query>    # Full pipeline: scan → retrieve → verify → load
memory_system.py staleness <dir>         # Age report
```

### Memory File Format

```yaml
---
name: finding_name
type: user | feedback | project | reference
verify_against: results.tsv | blackboard.md
claim: "the specific claim to verify"
---
Content (max 30 lines)
```

---

## Domain File Layout

```
domains/<domain>/
├── config.yaml              # Domain configuration
├── run.sh                   # Harness: takes config → outputs score
├── solve.py                 # (some domains) the code agents edit
│
├── program_static.md        # IMMUTABLE rules (read once by agents)
├── program.md               # DYNAMIC guidance (gardener rewrites)
│
├── blackboard.md            # Shared append-only state (agents write)
├── stoplight.md             # AUTO-GENERATED 30-line compressed state
├── recent_experiments.md    # AUTO-GENERATED last 5 experiments
├── meta-blackboard.md       # Meta-agent reflections
├── calibration.md           # Literature search (gen 1 only)
│
├── results.tsv              # All experiment results (append-only)
│   # format: id  score  keep/discard  "desc"  agent  design  time
│
├── best/                    # Current best configuration
│   ├── train.py (or config.yaml)
│   └── config_hash
│
├── workspace/               # EPHEMERAL (gitignored)
│   ├── agent0/train.py      # Agent 0's isolated copy
│   ├── agent1/train.py      # Agent 1's isolated copy
│   └── agentN/train.py
│
├── logs/
│   ├── agent0.jsonl          # Full Claude conversation trace
│   ├── agent1.jsonl
│   └── agentN.jsonl
│
├── memory/                   # Persistent domain memory (v4.7+)
│
├── DESIRES.md               # Agent telemetry: what they wish for
├── MISTAKES.md              # Agent telemetry: structured failures
├── LEARNINGS.md             # Agent telemetry: discovered facts
│
└── .nudge_data.json         # diagnose.py output for gardener
```

---

## End-to-End Data Flow

### One Experiment (micro)

```
Agent reads stoplight.md
  → forms hypothesis
  → edits workspace/agentN/train.py
  → bash run.sh exp-name "description" design_type
    → run.sh copies workspace config
    → runs training (GPU serialized via flock)
    → outputs score to stdout
  → agent appends result to results.tsv
  → agent appends finding to blackboard.md
  → agent updates MISTAKES.md / LEARNINGS.md / DESIRES.md
```

### One Monitor Cycle (meso)

```
[N minutes pass]
  → refresh_context.py regenerates stoplight.md + recent_experiments.md
  → meta-loop reads state → Claude reflection → new meta-blackboard.md
  → diagnose.py → trustloop_scorer → PQ score → decision
  → outer-loop acts on decision
```

### One Generation (macro)

```
1. Calibrate (gen 1 only) → calibration.md
2. Launch workers + meta-agent in screen sessions
3. Monitor loop (every N min): refresh → diagnose → act
4. Workers hit max turns or gardener stops them
5. Back up blackboard.md.genN
6. Apply decision (REDESIGN program.md / append taste.md lesson)
7. Launch next generation with updated scaffold
```

### Full Run (multi-generation)

```
Gen 1: calibrate → launch → monitor → REDESIGN (blind spots found)
  └─ program.md updated with new constraints
Gen 2: launch → monitor → NUDGE → NUDGE → NUDGE → escalate to REDESIGN
  └─ scaffold block diagnosed and fixed
Gen 3: launch → monitor → CONTINUE → CONTINUE → STOP_DONE
  └─ final meta-blackboard.md + lesson appended to taste.md
```

---

## MCP Servers

Optional Claude Code integration for domain inspection and trace analysis.

### rrma_mcp.py (Read-Only Domain Access)

**File:** `tools/rrma_mcp.py`

| Tool | Purpose |
|------|---------|
| `rrma_list_domains()` | List all domains |
| `rrma_domain_summary(domain)` | Quick overview (config, results count, best score) |
| `rrma_read_artifact(domain, type)` | Read: blackboard, program, results, experiments, desires, learnings, mistakes, calibration, config |
| `rrma_query_results(domain, filters)` | Grep results.tsv |
| `rrma_check_status(domain)` | Active screens, file mtimes, artifact freshness |

### trustloop_mcp.py (Trace Analysis)

**File:** `tools/trustloop_mcp.py`

| Tool | Purpose |
|------|---------|
| `trustloop_status()` | Overview: agent count, steps, thinking blocks, tool calls, experiments, best score |
| `trustloop_agent(id, mode)` | Per-agent: summary, thinking blocks, timeline |
| `trustloop_influence()` | Cross-agent influence analysis |
| `trustloop_compare(ids)` | Side-by-side agent comparison |

---

## Gardener Principles

**File:** `v4/taste.md`

These principles are learned from prior runs and consulted before every redesign decision. Updated automatically after each generation.

| # | Principle |
|---|-----------|
| 1 | **Less protocol = better science.** Plain blackboard beats structured CLAIM/RESPONSE. |
| 2 | **Config-tuning ≠ research.** High scores from parameter hacking → low PQ → stop. |
| 3 | **Simplification = maturity.** Dropping complexity + higher scores = understanding. |
| 4 | **Plateau = mapping the basin, not failure.** Long plateaus with high PQ mean agents are mapping the search space. |
| 5 | **Re-evaluate old failures post-breakthrough.** Context changes flip what works. |
| 6 | **Plan on stagnation, not round count.** Trigger replanning at < 0.5% improvement for 15+ experiments. |
| 7 | **Watch axis lock-in.** If all agents explore one dimension + flat scores, make others visible. |
| 8 | **Confirmation is a feature.** Multiple agents confirming the same thing = confidence. |
| 9 | **Low PQ + rising score = hacking.** Stop and force real research. |
| 10 | **High PQ + flat + no blind spots = done.** Search exhausted. |
| 11 | **High PQ + flat + blind spots = redesign.** Scaffold is blocking exploration. |

---

## Agent Telemetry (v4.4+)

Agents write three self-telemetry files that the gardener and TrustLoop read:

| File | Purpose | Format |
|------|---------|--------|
| `DESIRES.md` | Tools, context, or capabilities agents wish they had | Free-form |
| `MISTAKES.md` | Experiments that failed | Structured: what / result / lesson |
| `LEARNINGS.md` | Discovered facts about the environment | Free-form |

The TrustLoop scorer parses these for content (not just line counts) and feeds insights to the gardener's diagnosis.

---

## Hardware

| Machine | Specs | Role |
|---------|-------|------|
| **nigel.birs.ca** | RTX 4070 Ti SUPER 16GB, Ubuntu 24.04, torch 2.10.0+cu128 | GPU experiment execution |
| **Local Mac** | M2 Pro 32GB | MCP servers, scoring, monitoring |

### Running on nigel

```bash
ssh vincent@nigel.birs.ca
cd ~/researchRalph
bash v4/outer-loop.sh domains/<domain> 3 2 200 20
```

### Monitoring from Mac

```bash
ssh vincent@nigel.birs.ca "cat ~/researchRalph/domains/<domain>/results.tsv"
ssh vincent@nigel.birs.ca "screen -ls"
```

---

## Proven Results

| Version | Experiments | Hardware | Best Result | Domain |
|---------|-------------|----------|-------------|--------|
| v2 | 186 | 8x A100 | 1.048 BPB | GPT-2 |
| v3 | 135 | 1x RTX 4070 Ti | 0.9894 F1 (beat 0.97 ceiling) | SAE-bench |
| v4.0-4.2 | — | — | Hacking detection validated (PQ=6/30, STOP_HACKING fired) | — |
| v4.5 | 9 | 1x RTX 4070 Ti | 1.102 BPB | gpt2-tinystories-v44 |
| v4.7 | 231+ | — | 94.7% pass@many (MiniF2F) | rrma-lean |

### Key Domain Results

| Domain | Best Result | Key Finding |
|--------|-------------|-------------|
| rrma-lean | 94.7% pass@many (231/244 MiniF2F) | Mathlib steers proof strategy, not problem structure |
| sae-bench | 0.9894 F1 | 1-step ISTA + K-curriculum beats 5-step LISTA |
| gpt2-tinystories | 1.047 BPP | Throughput-over-capacity; overcomplication always lost |
| rrma-r1 | Rediscovered GRPO+PRM | Agents derived DeepSeek-R1 recipe from first principles |
| nirenberg-1d | 10x improvement | Agents invented Fourier spectral method autonomously |

**37 total domain directories** including 9 competitive games (battlebotgym-*), chaos experiments, and AF elicitation.

---

## v2 Legacy Components

Still present in the codebase for backwards compatibility:

| File | Purpose |
|------|---------|
| `core/launch.sh` | v2 launcher (git worktrees, 3 agent designs: vanilla/memory/blackboard) |
| `core/operator.sh` | v2 HITL controls: claim, request, direct, queue, ban, fact, hunch, strategy, pause, resume, repurpose |

---

## Version History

| Version | Date | Key Addition |
|---------|------|-------------|
| v2 | Mar 9 | Multi-agent blackboard |
| v3 | Mar 12 | Backward planning, stripped protocol |
| v4.0–4.2 | Mar 19–24 | Gardener + meta-loop (STOP_HACKING validated) |
| v4.3 | Mar 25–27 | DAG extraction, trace forensics, literature search |
| v4.4 | Mar 28 | Gardener reads DESIRES/MISTAKES/LEARNINGS |
| v4.5 | Mar 29–31 | TrustLoop scorer, stoplight, forensic pipeline |
| v4.6 | Mar 31 | Context optimization (81% fewer tokens) |
| v4.7 | Mar 31 | Agent-local workspaces, memory system, staleness checking |
| v4.8 | Apr 1 | Skeptical memory (verify claims against live sources) |
| v4.9 | Apr 2 | Single unified MCP server |
