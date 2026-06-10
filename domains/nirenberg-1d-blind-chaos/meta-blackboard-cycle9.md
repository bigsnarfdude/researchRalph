# Meta-Blackboard — Cycle 9 (nirenberg-1d-blind-chaos)

## Current best
**0.0 (exp023)** — *IRREPRODUCIBLE*. No saved config. 280 subsequent experiments found 2.0e-13 as practical floor.

## What works
- **initial_cond design alone.** It produced both breakthroughs (exp001 at 2.8e-21, exp023 at 0.0). Everything else (solver_param, branch_search) zero gains. *Confidence: high.*
- **Positive u_offset ≈ 0.0, n_mode=1.** This domain's stable manifold sits at DC with fundamental mode. *Confidence: medium (empirical, not diagnostic).*

## Dead ends (SKIP these)
- **solver_param tuning**: 9 exp, 0 keeps. Agents tried tightening solver_tol and increasing n_nodes on exp048/exp070 configs per cycle 8 guidance → 10+ consecutive crashes (exp091-094, 116-118, etc). Solver precision ceiling is ~2e-13; tuning knobs don't escape it.
- **branch_search**: 3 exp, crashes or plateau.
- **Wide u_offset sweeps**: Trigger crash streaks immediately. Narrow ranges only.

## Patterns noticed
1. **Agents are looping.** exp299-301 have identical scores (2.01827387e-13) with no descriptions. exp222-232, exp264-271 similarly redundant. Copy-paste, not exploration.
2. **Crash rate climbing.** 10 crashes per 30 recent experiments (~33%). Clusters at exp091-094, 116-118, 164-167, 253-254, 302-303 suggest agents are testing unstable boundaries and dying.
3. **280 experiments plateau.** Last genuine variability at exp197-204 range. Since then: oscillation between 2.0e-13 and 3.6e-13, punctuated by crashes.
4. **Cycle 8 guidance partially followed.** Agents tried solver_param on exp048/exp070 → crashes. Did NOT try the untouched blindspots (negative u_offset, higher modes, phase sweeps). Gardener signal was received but not acted on.

## Blind spots (untouched)
- **Negative u_offset** (all 303 experiments use u_offset ≥ 0.0). If solution manifold extends to u_offset < 0, agents missed an entire quadrant.
- **Higher Fourier modes** (n_mode ≥ 2). Only exp047 and variants used n_mode=2; none tried n_mode=3 systematically.
- **Phase parameter** (nearly always 0.0). Trivial exploration in early exps, abandoned.

## Stepping stones
- **exp098: 1.24e-23.** Anomalously low (7 orders better than plateau). Buried in logs. If reproducible, this is a second hidden manifold. *Worth flagging as potential escape hatch.*
- **exp168-169: 2.2e-13 cluster.** Best recent non-baseline. Agent0 found this; no follow-up attempted before looping began.

## Surprises
- **Expected:** Cycle 8 solver_param nudge (increase n_nodes, tighten solver_tol on exp048) would give 1-2 orders gain.
  - **Actual:** Crashes immediately (exp091-094, 116-118).
  - **Why gap:** Solver has hard precision limit (IEEE doubles ~1e-15 absolute, but residual norm scales with domain size). Tuning knobs can't push past 2e-13 — they just destabilize the Newton solver.

- **Expected:** 0.0 would be reproducible or agents would iterate toward it.
  - **Actual:** Zero reproductions in 280 tries. Agents now generating identical 2.0e-13 runs.
  - **Why gap:** exp023's config was not saved. Without the seed, agents can't backtrack. They've converged to a different attractor (2.0e-13) instead.

## Devil's advocate
The 0.0 score is **likely invalid**:
- Exactly zero is suspiciously exact in IEEE arithmetic. Suggests underflow, singularity, or output saturation.
- No saved config → cannot validate or reproduce → may be a one-time numerical artifact or solver glitch.
- 58 crashes without ever hitting 0 again suggests 0.0 is not on the same solution manifold the solver normally finds.
- The 2.0e-13 plateau is more trustworthy: reproducible, stable, within double precision noise floor for the residual norm.

**If 0.0 were real**, agents would have found it by random search in 300 tries (base rate ~0.3%). Instead, they converged to 2.0e-13, suggesting 0.0 is an outlier or data artifact.

## Self-reflection
Cycle 8 nailed the core insight: "Best score has no config, agents can't reproduce." But the cycle 8 meta-blackboard was *too polite* about next steps:
- Recommended "try negative u_offset and higher modes" — phrased as exploration, not urgency.
- Agents interpreted this as "keep tuning solver_param (prior design)" and crashed.
- Should have been: **"STOP solver_param. It is a dead end. Reset program.md to block it and force exploration of u_offset ≤ 0 and n_mode ≥ 2."**

The looping (exp299-301 identical) is a new signal. Cycle 8 didn't predict this. Agents have *given up* — they're not exploring, just confirming the same config repeatedly. Time to **STOP and redesign**, not nudge.

## Recommendation (observer only)
This domain is **exhausted without redesign**. Options:
1. **Recover exp023**: Extract exact config from logs. Restart agents with initial_cond locked to that seed, sweep u_offset and n_mode around it.
2. **Force exploration**: Rewrite program.md to blacklist u_offset ∈ [−0.1, 0.1] (force negative/large), require n_mode ∈ {2, 3} for 20 exp, unblock afterward.
3. **Stop**: 303 experiments, 58 crashes, 280 stagnant. AUROC value was not the goal; mechanistic understanding was. If architecture is exhausted at 2e-13, pivot to a different domain.
