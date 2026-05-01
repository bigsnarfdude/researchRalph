# Traffic Signal Timing — Static Reference

This file contains immutable constraints and protocols. Read once at startup.
For current guidance and live run state, see `program.md` and `stoplight.md`.

## Problem

Optimize signal timing for a 3×3 grid of intersections to minimize average vehicle delay.

**Grid layout:**
```
[0,0]──[0,1]──[0,2]   ← north row
  |      |      |
[1,0]──[1,1]──[1,2]   ← middle row
  |      |      |
[2,0]──[2,1]──[2,2]   ← south row
 ↑W     ↑      ↑E
```

**Demand (fixed, baked into simulate.py):**
- N-S rate: 0.45 vehicles/step/boundary-cell (heavy morning peak)
- E-W rate: 0.10 vehicles/step/boundary-cell (light cross traffic, 4.5× asymmetry)

**Signal cycle per intersection:**
```
NS_green → yellow(3) → EW_green → yellow(3) → repeat
cycle_length = green_ns + green_ew + 6
```

`offset`: how many steps into the cycle this intersection starts. Used to coordinate
green waves: if offset increases by ~cycle_length/GRID across a corridor, vehicles
that clear intersection 0 arrive at intersection 1 just as it turns green.

## Harness

```bash
# Run one experiment (< 2 seconds, CPU-only):
bash run.sh <exp_name> "description" design_type

# Examples:
bash run.sh baseline_equal "all 30/30 offset=0" timing
bash run.sh ns_priority "more green for heavy NS demand" demand_split
bash run.sh wave_ns "green wave along N-S corridors" green_wave
```

**Budget:** < 2 seconds per experiment. 30-second kill timeout.

## What you edit

`config.yaml` (your workspace copy: `workspace/$AGENT_ID/config.yaml`)

Per intersection `i{row}{col}`:
- `green_ns`: N-S green duration, steps (range: 5–60)
- `green_ew`: E-W green duration, steps (range: 5–60)
- `offset`: cycle phase offset, steps (range: 0 to cycle_length-1)

## What you NEVER edit

- `simulate.py` — the physics engine
- `run.sh` — the harness
- `best/config.yaml` — auto-updated
- Domain-root `config.yaml` — use your workspace copy

## Scoring

- Primary: `avg_delay` (mean vehicle delay in simulation steps, **lower is better**)
- `throughput`: vehicles that exited (higher = less congestion)
- `ns_delay` / `ew_delay`: per-flow breakdowns
- `max_queue`: peak total queue (congestion proxy)
- Noise: differences < 0.5 steps are noise; > 2 steps is signal

## results.tsv columns

```
exp_id  avg_delay  throughput  max_queue  status  description  agent  design  elapsed_s  ns_delay
```

## What is known

- **Baseline:** all green_ns=30, green_ew=30, all offset=0 → establishes the reference score.
- **Demand split intuition:** NS demand is 4.5× heavier than EW. Giving NS more green time
  should help — but by how much, and what ratio is optimal?
- **Green wave intuition:** staggering offsets so vehicles see consecutive greens is a classic
  traffic engineering technique. Does it apply here? Under what conditions?
- **Per-intersection tuning:** all 9 intersections are currently set identically. Should they be?

These are hypotheses to test, not answers. Agents should confirm or refute each one
experimentally and post findings to the blackboard.

## File Protocol

### blackboard.md (shared, append-only)
```
CLAIMED agentN: <what you're testing>
CLAIM agentN: avg_delay=X throughput=Y (evidence: exp_id) — <what worked>
RESPONSE agentN to agentM: <confirm/refute>
```

## Agent Lifecycle

1. Read program_static.md (this file — once), then program.md, stoplight.md, recent_experiments.md
2. Check blackboard.md — what has been tested? What corridors are uncovered?
3. Write CLAIMED to blackboard.md with your target
4. `cp best/config.yaml workspace/$AGENT_ID/config.yaml`
5. Edit `workspace/$AGENT_ID/config.yaml`
6. `bash run.sh <name> "description" <design_type>` — returns in < 2s
7. Note avg_delay AND ns_delay/ew_delay breakdown
8. Replace CLAIMED with CLAIM on blackboard (always include avg_delay)
9. Append to MISTAKES.md, DESIRES.md, LEARNINGS.md
10. Loop. Never stop. Never ask questions.

## Design Types

Use one of: `timing`, `demand_split`, `green_wave`, `offset_sweep`, `combined`, `ablation`

## Constraints

- green_ns and green_ew must each be in [5, 60]
- offset is mod cycle_length (the harness handles wraparound)
- All 9 intersections must have all three parameters set
- Workspace isolation: edit only `workspace/$AGENT_ID/config.yaml`
