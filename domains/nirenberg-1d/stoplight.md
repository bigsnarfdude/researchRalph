# Stoplight — nirenberg-1d
Status: STAGNANT | Best: nan (exp001) | Experiments: 666 | Stagnation: 665 since last breakthrough

## Agents
- agent0: 348 exp, 0 breakthroughs, rate 19%, best nan
- agent1: 317 exp, 0 breakthroughs, rate 44%, best nan
- manual: 1 exp, 1 breakthroughs, rate 100%, best nan

## Alerts
- crash_streak: Agent has 3 consecutive crashes — likely broken config or OOM
- crash_streak: Agent has 3 consecutive crashes — likely broken config or OOM
- crash_streak: Agent has 3 consecutive crashes — likely broken config or OOM

## Recent blackboard (last 20 entries)
Shared lab notebook. Write what you tried, what happened, and why.
Read before starting to avoid duplicating work.
## Previous generation summary
The previous generation's findings are in meta-blackboard.md. Read it.
CLAIMED agent0: Branch boundary mapping — systematically sweep u_offset to find exact transition points between trivial↔positive and trivial↔negative convergence basins. Also: push Fourier solver residuals on each branch.
CLAIM agent0: Basin boundary at |u_offset| ≈ 0.463 (Fourier N=16, amp=0) — branch=[boundary mapping]
  - |u_offset| < 0.463 → trivial branch (mean≈0)
  - |u_offset| > 0.463 → nontrivial branch (mean≈±1)
  - Near boundary (0.463-0.470): FRACTAL basin structure — solver oscillates between positive and negative branches chaotically
  - Structure is Z₂ symmetric: positive side boundary ≈ negative side boundary
  - Evidence: exp533-exp566 boundary sweeps
CLAIM agent0: Fourier N=16 residuals per branch — branch=[all three]
  - Trivial: residual=0.0 (exact zero, mean=0.0) — exp532
  - Positive: residual=3.9e-15 (mean=+1.000019) — exp529
  - Negative: residual=9.8e-14 (mean=-1.000019) — exp531
