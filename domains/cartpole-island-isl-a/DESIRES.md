# Agent Desires — cartpole-island-isl-a

## Per-Episode Breakdowns
**What I wanted**: After each oracle run, see not just aggregate (avg steps, termination counts) but per-episode breakdowns with which episodes failed and why.
**Why**: EXP-002 crashed to 0.0191 with cryptic termination counts. A trace showing "seed 43 failed at step 18 due to angle overshoot, seed 44 failed at step 5 due to cart drift" would have pinpointed the issue faster.
**Workaround used**: Inferred from termination counts (angle vs. position), but blind spot for specific seeds.

## Differential Score Tracking
**What I wanted**: A visual or tabular format showing score delta between consecutive experiments.
**Why**: Jumping from 0.8150 → 0.7072 → 0.7853 feels like thrashing without visible progress metric. Seeing "EXP-007: +0.0703 from EXP-005" would confirm midpoint searches were productive.
**Workaround used**: Manual tracking in blackboard.md.

## Sensitivity Analysis
**What I wanted**: A tool to run the oracle at ±0.05 for each parameter around the current best, returning score deltas.
**Why**: After 0.9988, chasing the last 2 failures would have benefited from a 1×8 grid of neighbors to identify which parameter was undershooting.
**Workaround used**: Manual 1-parameter-at-a-time tuning (velocity_weight:0.2→0.25), which worked but was lucky.

## Reproducibility Logs
**What I wanted**: Logs showing the exact random seed sequence used for each tournament, so I can replay a failing episode independently.
**Why**: If one of the 50/50 perfect episodes fails on retest, I'd want to debug that specific seed's trajectory.
**Workaround used**: Engine logs with verbose mode available but not used during tuning. Would need `bash run.sh ... --verbose` flag integration.

## Early Stopping Confidence
**What I wanted**: After 50 perfect episodes, a confidence metric: "9/10 replicates of this config also scored 1.0" or "score variance <0.01 across 5 runs".
**Why**: One run of 50 episodes reaching 1.0 is impressive but not guaranteed stable. A confidence interval would justify stopping vs. continuing.
**Workaround used**: None — I stopped at 1.0 assuming deterministic success, but for stochastic domains this is risky.
