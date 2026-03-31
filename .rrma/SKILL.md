---
description: >
  Use this skill when managing RRMA (ResearchRalph Multi-Agent) runs:
  listing domains, checking run status, reading experiment results,
  inspecting shared artifacts (blackboard, program, desires, learnings,
  mistakes), or configuring domains. Triggers on: RRMA, domain status,
  experiment results, agent runs, blackboard, ResearchRalph, multi-agent
  control, run management.
---

# RRMA Run Management

You have ResearchRalph MCP tools for managing multi-agent research runs.

## Domain concepts

**RRMA** (ResearchRalph Multi-Agent) runs multiple Claude-powered agents
in parallel on domain experiments. Each domain has a harness (run.sh)
that evaluates agent work and returns a score.

**Domains** are directories under `domains/`. Each contains:

- `config.yaml` — domain settings (metric, target score, model, etc.)
- `program.md` — instructions agents follow
- `run.sh` — harness that evaluates experiments and returns scores
- `blackboard.md` — shared coordination space (read/write by all agents)
- `results.tsv` — experiment registry with scores (append-only)
- `DESIRES.md` — agent requests to the gardener
- `LEARNINGS.md` — things that worked, shared across agents
- `MISTAKES.md` — things that didn't work, shared across agents
- `calibration.md` — baseline performance from calibration step
- `logs/` — agent trace logs (JSONL)

**Runs** are managed by `v4/outer-loop.sh` which:
1. Calibrates baseline performance
2. Launches N agents as parallel Claude Code sessions (screen)
3. Monitors progress and diagnoses quality
4. Rotates agent sessions when they stall
5. Stops when done or when quality drops

**Artifacts** are files agents read/write during a run. The blackboard
is the primary coordination mechanism. Results.tsv tracks experiments.

## Available tools

- `rrma_domains` — list all domains with experiment counts and best scores
- `rrma_config` — read a domain's configuration and available artifacts
- `rrma_status` — check if a run is active (screen sessions) + latest scores
- `rrma_results` — full experiment history with per-agent breakdowns
- `rrma_artifact` — read any shared artifact content
- `rrma_artifacts_summary` — overview of all artifacts without full content

## When to use which tool

- "What domains exist?" → `rrma_domains`
- "What's the config for X?" → `rrma_config`
- "Is anything running?" → `rrma_status`
- "What's the best score?" → `rrma_results`
- "What's on the blackboard?" → `rrma_artifact(domain, "blackboard")`
- "Show me all artifacts" → `rrma_artifacts_summary`

## ResearchRalph vs TrustLoop

- **ResearchRalph MCP** = run control (what's running, what scored, what's configured)
- **TrustLoop MCP** = forensics (what agents thought, influence chains, step-by-step replay)

Use ResearchRalph to check current state. Use TrustLoop to investigate why.
