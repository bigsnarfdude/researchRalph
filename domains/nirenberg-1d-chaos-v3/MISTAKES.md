
## agent3 — Mistakes

- exp007: Used mode-2 perturbation but it didn't improve residual over mode-1. IC oscillation mode is irrelevant — solver converges to same branch attractor.
- exp016: Optimized at n_nodes=200 when agent1 had already shown n_nodes=300 is 3.4x better. Should read blackboard before running.

## agent2 — Mistakes

- exp010 vs exp009: Increased n_nodes from 200→300 for negligible gain (8.82e-11→8.59e-11). Waste of an experiment — should have tried a different approach instead.
- exp012: Changed u_offset and amplitude simultaneously — should test one variable at a time for clean attribution.
- exp030: First Fourier solver attempt crashed because newton_tol=1e-14 was too tight for 64 modes to converge in 50 iterations. Should have started with relaxed tol.
- exp042: 128 Fourier modes crashed — larger system makes Newton harder. Should have tried FEWER modes first (which turned out to be better).
- Spent too many experiments on scipy parameter tuning (nodes, tol) before discovering the Fourier solver option in solve.py. Should read the solver code earlier.

## agent0 — Mistakes

- exp018: Ran n_nodes=300 positive branch expecting improvement but got 8.15e-11 (worse than agent1's 2.60e-11). Didn't account for possible differences in how best/config.yaml evolved between runs.
- exp022: Tested flat IC (amp=0) for positive branch — confirmed IC details don't matter (same as agent2's finding). Partially redundant.
- Should have started basin boundary exploration earlier — this was the most novel contribution.

## agent1 — Mistakes

- exp019: Tested flat IC (amp=0) when agent2 had already established that IC details are irrelevant. Redundant experiment.
- exp029: Crash at u_offset=0.52 — could have predicted solver failure near basin boundary. Should approach boundaries more cautiously with smaller steps.
- exp073/074: Fourier solver with newton_tol=1e-12 gave only 9.34e-14 — should have immediately tried tighter tolerance (1e-14 gave 1.86e-16, a 500x improvement). Agent2's results should have hinted at this.
- exp066/068: Spent two experiments searching for higher-mode branches that don't exist. Could have checked the equation structure first — K_frequency=1 means only 3 branches.
