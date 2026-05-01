# Traffic Signal Timing — Dynamic Guidance

Current regime: **EXPLORATION** — no results yet. Start with baselines.

## Suggested first experiments

1. `baseline_equal` — all green_ns=30, green_ew=30, all offset=0
   - Establishes the reference score. Run this first.

2. `ns_priority_v1` — increase green_ns to ~40, decrease green_ew proportionally
   - NS demand is 4.5× EW. Does more NS green time help, and by how much?

3. `offset_sweep` — keep best timing, try varying offsets (e.g., +10 per row)
   - Classic green wave. Does it help? Confirm or refute on blackboard.

## Open questions

- What is the actual baseline avg_delay?
- What green_ns / green_ew ratio minimizes avg_delay?
- Do staggered offsets (green waves) help or hurt for bidirectional traffic?
- Should all 9 intersections have the same timing, or should they differ?

## Closed brackets (dead ends)

None yet.

## Current best

No experiments run yet. First agent to run baseline establishes the reference.
