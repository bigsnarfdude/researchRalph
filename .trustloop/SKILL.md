---
description: >
  Use this skill when investigating RRMA (ResearchRalph Multi-Agent)
  runs, analyzing agent behavior, inspecting experiment traces,
  querying blackboard or artifact contents, following cross-agent
  influence chains, or debugging multi-agent research outcomes.
  Triggers on: trace analysis, agent investigation, experiment
  forensics, blackboard inspection, multi-agent debugging, RRMA,
  TrustLoop, swarm analysis, harness scores.
---

# RRMA Trace Investigation

You have TrustLoop MCP tools for investigating multi-agent research runs.

## Domain concepts

**RRMA** (ResearchRalph Multi-Agent) runs multiple Claude-powered agents
in parallel on domain experiments. Agents coordinate through shared
artifacts and iterate toward a score on a harness evaluator.

**Agents** run in sessions. Each session produces a trace: a sequence
of steps where each step has reasoning (thinking), visible output,
tool calls, and artifact interactions.

**Artifacts** are the shared coordination layer:

- `program` — task instructions and scaffold (read-heavy, rarely written)
- `blackboard` — free-form coordination space (read and write by all agents)
- `results` — experiment registry with scores (append-only)
- `desires` — agent requests to the gardener/coordinator
- `learnings` — things that worked, shared across agents
- `mistakes` — things that didn't work, shared across agents

**Influences** are directed edges in a dependency graph. When agent B
reads an artifact that agent A previously wrote to, an influence edge
is recorded: A -> B via that artifact. These chains show how ideas
propagate through the swarm.

**Thinking blocks** (`reasoning_content`) contain the agent's extended
thinking before each action. This is the primary evidence for
understanding *why* an agent made a decision.

## Available tools

- `trustloop_status` — run overview: agents, sessions, steps, scores, artifacts
- `trustloop_agent` — agent detail with three modes: summary, thinking, timeline
- `trustloop_search` — full-text search across all traces
- `trustloop_artifact` — shared artifact content and metadata
- `trustloop_influences` — cross-agent influence edges, optionally filtered
- `trustloop_step` — raw step data for a specific trace and step index
- `trustloop_index` — compact text index of the entire run

## Investigation workflow

1. Start with `trustloop_status` to understand the run shape
2. Use `trustloop_influences` to see cross-agent dependency patterns
3. Drill into agents: summary first, then thinking or timeline for detail
4. Use `trustloop_search` for keyword evidence (tactic names, error messages, artifact references)
5. Check `trustloop_artifact` for the current state of shared artifacts
6. Use `trustloop_step` when you need the exact raw data for a specific step

## When answering

- Cite specific agent IDs and step numbers
- Quote thinking blocks to explain reasoning
- Reference artifact reads/writes to show coordination
- Note influence chains when explaining cross-agent effects
- Flag discrepancies (e.g., scores in trace steps but not in results.tsv)
- Distinguish between formally registered experiments and informal harness runs
