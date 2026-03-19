# RRMA v4: Self-Recursive Research Meta-Agent

The outer loop that redesigned v1→v2→v3 was human. v4 automates it.

## What it does

Wraps the multi-agent research loop with a **gardener** — an outer agent that:
1. **Scores process quality** from artifacts (papers cited, ablations run, architecture diversity)
2. **Stops runs** when agents are hacking (config-tuning, not researching) or exhausted
3. **Redesigns the scaffold** (rewrites program.md) when agents are stuck

## Quick start

```bash
# Launch v4 on a domain (outer loop handles everything)
bash v4/outer-loop.sh domains/sae-bench 5 4 200 20
#                      domain            gens agents turns monitor_min
```

The gardener calibrates, launches workers + meta-agent, monitors every 20 min,
and stops/redesigns/continues based on process quality scoring.

## Architecture

```
outer-loop.sh (the gardener — stop authority, scaffold editing)
  ├── calibrate.sh          pre-run literature search
  ├── launch-agents.sh      start N workers + meta-agent in screen sessions
  │   ├── worker 0..N       claude -p --dangerously-skip-permissions
  │   └── meta-loop.sh      sleep/compress/reflect every 30 min
  ├── diagnose.sh           process quality scoring (0-30) every check
  ├── taste.md              inherited + learned principles
  └── generate-meta-blackboard.sh   post-run distillation
```

## Stopping rules

| process_quality | score_trend | blind_spots | action |
|---|---|---|---|
| LOW (< 10/30) | any | any | **STOP_HACKING** |
| HIGH | improving | any | **CONTINUE** |
| HIGH | flat 15+ exp | nonempty | **REDESIGN** |
| HIGH | flat 15+ exp | empty | **STOP_DONE** |

## Process quality scoring (0-30)

Measured from blackboard.md, sae.py, results.tsv:
- Papers/references cited
- Architecture classes created
- Explanatory reasoning (because/why/mechanism)
- Ablation experiments
- Simplification moves
- Unique experiment designs

## taste.md

The gardener's judgment, seeded by human experience and updated automatically.
After each generation, the gardener appends lessons learned. Over time, the
system builds its own taste.

## Files

```
v4/
├── outer-loop.sh              generation loop (calibrate → launch → monitor → stop/redesign)
├── diagnose.sh                process quality → CONTINUE/STOP_HACKING/STOP_DONE/REDESIGN
├── calibrate.sh               literature search via claude -p
├── taste.md                   inherited principles (human-seeded, auto-updated)
├── launch-agents.sh           start workers + meta-agent in screen sessions
├── stop-agents.sh             kill all rrma sessions
├── meta-loop.sh               meta-agent sleep/compress/reflect cycle
├── generate-meta-blackboard.sh  post-run distillation
└── env.sh                     portable claude CLI PATH detection
```

## Comparison to core/

`core/launch.sh` is v2 — multi-agent with blackboard, worktrees, memory files.
`v4/outer-loop.sh` is the self-recursive layer on top. It uses its own simpler
launcher (v3-style: plain blackboard, no roles) and adds the gardener.

Use `core/` when you want structured multi-agent with operator control.
Use `v4/` when you want fully autonomous research with self-correction.
