# researchRalph — Claude Code Instructions

## What This Is

A multi-agent research framework with three operating modes:

- **v2** (`core/`): Structured multi-agent with blackboard, memory, operator control
- **v3** (simplified): Plain blackboard, no roles, Ralph Wiggum loops
- **v4** (`v4/`): Self-recursive — adds a gardener (outer agent) with stop authority and scaffold editing

## Key Commands

```bash
# v2: Multi-agent with operator control
./core/launch.sh domains/<domain> N [--gpu]
./core/operator.sh domains/<domain> ban "dead end description"

# v4: Fully autonomous (gardener handles everything)
bash v4/outer-loop.sh domains/<domain> [max_gens] [num_agents] [max_turns] [monitor_min]
```

## Adding a Domain

```bash
cp -r domains/template domains/my-domain
# Edit: config.yaml (or prompt_config.yaml), run.sh, program.md
```

A domain needs: something agents edit, a harness that outputs a score, and instructions.

## v4 Architecture

```
outer-loop.sh → calibrate.sh → launch-agents.sh → [monitor loop] → diagnose.sh
                                    ↓
                              workers + meta-agent (screen sessions)
                              meta-loop.sh compresses blackboard every 30 min
```

Process quality scoring (0-30) from artifacts. Stopping rules:
- PQ < 10 after 15 exp → STOP_HACKING (rewrite program.md)
- PQ ≥ 10 + flat + no blind spots → STOP_DONE
- PQ ≥ 10 + flat + blind spots → REDESIGN

## Proven Results

- v2: 186 experiments, 8×A100, 64% hit rate, 1.048 BPB (GPT-2)
- v3: 135 experiments, 1×RTX 4070 Ti, 0.9894 F1 (SAE-bench, beat 0.97 ceiling)
- v4: Hacking detection validated (PQ=6/30, STOP_HACKING fired correctly)

## Hardware

Single homogeneous GPU node required for ML domains. CPU-only works for prompt optimization.

## Critical Files

- `v4/taste.md` — The gardener's principles. Human-seeded, auto-updated.
- `v4/diagnose.sh` — Process quality scoring. Currently tuned for code-editing domains.
- Domain `program.md` — Agent instructions. The gardener can rewrite this.
- Domain `blackboard.md` — Shared state. Append-only.
