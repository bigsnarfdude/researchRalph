# Stoplight — nirenberg-1d-chaos
Status: STAGNANT | Best: nan (exp001) | Experiments: 72 | Stagnation: 71 since last breakthrough

## Dead ends — do NOT retry
- Design 'perturbation' has 3 experiments, 0 keeps — abandon this approach

## Gaps — unexplored
- 2 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 44 exp, 1 breakthroughs, rate 9%, best nan
- agent1: 28 exp, 0 breakthroughs, rate 7%, best nan

## Alerts
- crash_streak: Agent has 3 consecutive crashes — likely broken config or OOM
- deep_stagnation: No improvement in 71 experiments — search space may be exhausted or agents are stuck

## Recent blackboard (last 20 entries)
Shared lab notebook. Write what you tried, what happened, and why.
Read before starting to avoid duplicating work.
## Previous generation summary
The previous generation's findings are in meta-blackboard.md. Read it.
CLAIMED agent1: Testing higher Fourier modes (80, 96, 128) with loose Newton tol to push non-trivial branch residuals below 3e-13. Also exploring mode-2/mode-3 IC to find possible 4th branch solutions.
CLAIM agent0: Residual floor confirmed — 64 modes: ~3e-13 RMS for ±1 branches (u³ aliasing). 128 modes: 1.3e-12 (conditioning worse). newton_tol=1e-13 crashes. Trivial: 7.6e-23 (exact). (exp048-053)
CLAIMED agent0: Exploring whether higher-mode oscillatory solutions exist — mode-2/3 ICs with various u_offset values. Also investigating the norm≈0.07 near-bifurcation solution seen in exp044/046.
