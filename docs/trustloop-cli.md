# TrustLoop — Forensic Analysis for Multi-Agent Runs

Converts RRMA agent logs into structured traces (OpenTraces schema + multi-agent extensions), then lets you interrogate agent behavior through natural language forensic queries.

## Quick Start

```bash
# 1. Ingest: convert raw RRMA logs → structured traces
./trustloop ingest domains/rrma-lean --logs data/rrma_lean_logs

# 2. Launch forensic session
./trustloop chat

# 3. Ask anything
trustloop> which agent was most productive?
trustloop> what did agent3 struggle with?
trustloop> did information flow between agents or did they work in isolation?
trustloop> search for "omega" across all traces
```

## Architecture

```
RRMA domain run
  ├── logs/agent*.jsonl          Raw Claude Code conversation logs
  ├── blackboard.md              Shared agent state
  ├── meta-blackboard.md         Compressed cross-generation memory
  ├── results.tsv                Scored experiments
  ├── program.md                 Agent instructions
  ├── DESIRES.md                 Agent tool/context requests
  ├── LEARNINGS.md               Discovered knowledge
  └── MISTAKES.md                Failure patterns
        │
        ▼
tools/rrma_traces.py             Converter: raw logs → OpenTraces schema
        │
        ▼
/tmp/rrma_traces.jsonl           1 SwarmRecord + N TraceRecords
        │
        ▼
tools/trace_forensics.py         TraceStore: index + 6 query methods
        │
        ├──→ trustloop CLI       11 commands, claude -p for forensic asks
        ├──→ trustloop_server.py REST API + web UI
        └──→ claude -p           Direct pipe for ad-hoc queries
```

## Schema: OpenTraces v0.1.0 + RRMA Extensions

Base schema adopted from [opentraces](http://opentraces.ai). Six additive models for multi-agent coordination:

| Model | Purpose |
|-------|---------|
| `SwarmRecord` | Wraps N TraceRecords from one coordinated run |
| `SharedArtifact` | Versioned blackboard, results, program.md snapshots |
| `SupervisoryDecision` | Gardener PQ scores + STOP/NUDGE/REDESIGN actions |
| `ExperimentResult` | Many scored experiments per agent session |
| `CrossAgentInfluence` | Causal provenance: who wrote what, who read it |
| `ScaffoldVersion` | program.md evolution across generations |

Four optional fields added to `Step`: `swarm_agent_id`, `artifact_reads`, `artifact_writes`, `experiment_result`. Zero breaking changes to base schema.

## CLI Commands

### Ingest & Status

```bash
./trustloop ingest domains/rrma-lean --logs data/rrma_lean_logs
./trustloop status
```

### Forensic Queries (via claude -p)

```bash
./trustloop ask "which agent was most productive?"
./trustloop ask "what did agent3 struggle with?"
./trustloop ask "did any agent game the results?"
./trustloop chat                                    # interactive
```

### Direct Queries (instant, no API)

```bash
./trustloop search omega                            # full-text search
./trustloop search "ring tactic"
./trustloop agent agent3                            # summary
./trustloop agent agent3 --thinking                 # reasoning blocks
./trustloop agent agent0 --timeline                 # action chronology
./trustloop artifact blackboard                     # shared state content
./trustloop artifact program
./trustloop influences                              # all cross-agent flows
./trustloop influences agent4                       # filtered
./trustloop step 2acc1f71 42                        # specific step detail
./trustloop index                                   # compact forensic index
```

### Web Server

```bash
./trustloop serve --port 8765
# http://localhost:8765/trustloop_viewer.html
# REST API: /api/traces/search?q=omega
#           /api/traces/thinking?agent_id=agent3
#           /api/traces/artifact?type=blackboard
#           /api/traces/influences
#           /api/status
```

## What Forensic Queries Find

Tested on real rrma-lean data (30 traces, 8 agents, 13,930 steps, 1,193 thinking blocks):

| Query | Finding |
|-------|---------|
| "what did agent3 struggle with?" | omega can't handle exponentials — same failure 3 times, never systematized a fix |
| "did information flow between agents?" | Real-time concurrent coupling — agent0 saw agent6's score mid-session |
| "how did agents coordinate?" | Broadcast not relay — meta-blackboard caused convergent planning across all agents |
| "which agent wrote to blackboard most?" | agent4 = swarm bookkeeper — verification tables, timeout false-negative diagnosis |
| "ring tactic discovery?" | Null result — seeded knowledge, not emergent. Real finding: swarm stalled after EXP-001 |
| "most surprising finding?" | Failures are silent type errors in complete-looking proofs, not missing sorry stubs |

## Files

```
trustloop                        CLI entry point (this tool)
trustloop_server.py              Web server + REST API
trustloop_viewer.html            Web UI (domain explorer + chat)
tools/
  rrma_traces.py                 OpenTraces schema + converter
  trace_forensics.py             TraceStore + query engine + Claude tool-use loop
  forensic.sh                    claude -p wrapper (shell version)
docs/
  trustloop-cli.md               This file
  opentraces-rrma-schema-comparison.md   Schema extension proposal
```

## Data Sizes (rrma-lean)

| Metric | Value |
|--------|-------|
| Raw logs | 30 JSONL files, ~200 MB |
| Traces JSONL | 17 MB (31 records) |
| Forensic index | 3.9 KB (fits any context window) |
| Steps | 13,930 |
| Thinking blocks | 1,193 |
| Tool calls | 5,330 |
| Cross-agent influences | 56 |
| Shared artifacts | 3 (blackboard, results, program) |
