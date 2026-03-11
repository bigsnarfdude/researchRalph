# researchRalph v2 — Claude Code Instructions

## What This Is

A domain-agnostic multi-agent research framework. Agents collaborate through a shared blackboard to optimize any system with editable parameters and a scalar objective.

## Structure

```
core/           — Framework scripts (launch, stop, monitor, conductor, collect)
domains/        — Reference implementations + template for new domains
docs/           — Architecture, cognitive designs, extending, security
```

## Key Patterns

### Single Agent
```bash
./core/run-single.sh domains/<domain>
```
Loop: read state → pick experiment → run → record → update state → repeat.

### Multi-Agent (Blackboard)
```bash
./core/launch.sh domains/<domain> N [--gpu]
```
N agents in git worktrees, shared blackboard.md + results.tsv + strategy.md.

### Conductor (Reactive)
```bash
./core/conductor.sh domains/<domain>
```
Watches blackboard for REQUEST lines, spawns ephemeral agents.

## Adding a Domain

```bash
cp -r domains/template domains/my-domain
# Edit: config.yaml, run.sh, program.md
./core/launch.sh domains/my-domain 4
```

## Hardware Constraint

RRMA (researchRalph Multi-Agent) requires a **single homogeneous GPU node** (e.g., 8×A100 or 8×H100). Mixed GPU types across nodes cause errors due to hardware speed/memory differences producing incomparable scores and race conditions. Do not use multi-machine deployments with different GPU types.

## Proven Results

- 186 experiments, 8×A100, GPT-2 TinyStories
- Blackboard design: 64% hit rate, 1.048 BPB
- Vanilla (no memory): 17% hit rate, 1.152 BPB
- Key: structured memory (facts/failures/hunches) prevents repeating dead ends
