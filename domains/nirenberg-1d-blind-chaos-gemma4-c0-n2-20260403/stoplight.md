# Stoplight — nirenberg-1d-blind-chaos-gemma4-c0-n2-20260403
Status: STAGNANT | Best: 0.0 (exp001) | Experiments: 153 | Stagnation: 152 since last breakthrough

## Dead ends — do NOT retry
- Design 'initial_cond' has 107 experiments, 0 keeps — abandon this approach

## Agents
- agent0: 78 exp, 1 breakthroughs, rate 1%, best 0.0
- agent1: 75 exp, 0 breakthroughs, rate 0%, best 0.0

## Alerts
- crash_streak: Agent has 3 consecutive crashes — likely broken config or OOM
- crash_streak: Agent has 3 consecutive crashes — likely broken config or OOM
- crash_streak: Agent has 3 consecutive crashes — likely broken config or OOM

## Recent blackboard (last 20 entries)
- The residual limit seems related to the Newton tolerance (1e-12).
---
**[agent1, 17:10]**
CLAIM: The non-trivial solution is highly robust and its residual is largely determined by the Newton tolerance and mesh density, showing little sensitivity to the initial guess parameters (u_offset, amplitude, n_mode, phase) once the branch is found.
EVIDENCE: Repeated experiments (exp032 to exp040) with different initial guesses all converged to the same solution (norm ~1.001296) with residuals in the 10^-13 range.
---
**[agent1, 17:19]**
CLAIM: The Nirenberg 1D problem has at least three distinct solution branches: 
1. Trivial branch: norm 0.0, residual ~10^-20.
2. Large branch: norm ~1.001, residual ~2e-13.
3. Small branch: norm ~0.0705, residual ~2e-14.
The small branch (norm ~0.07) provides a significantly lower residual than the large branch. 
The residual for the small branch is limited by the mesh resolution (n_nodes=300) and Fourier modes (64), and appears insensitive to further increases in Newton tolerance or iterations once a certain threshold is reached.
---
**[agent1, 17:25]**
CLAIM: The small solution branch (norm ~0.07) is a real branch with a residual floor around 1.7e-14.
EVIDENCE: 
- Increasing Newton tolerance below 1e-14 leads to crashes (exp138).
- Increasing solver tolerance or changing parameters (u_offset, amplitude, n_mode, phase) does not significantly change the residual once the branch is found.
- The residual seems to be fundamentally limited by the solver's precision/convergence properties for this specific solution branch.
