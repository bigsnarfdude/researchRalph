# Stoplight — nirenberg-1d-chaos-v3-haiku
Status: STAGNANT | Best: 1.13671782e-23 (exp058) | Experiments: 125 | Stagnation: 64 since last breakthrough

## What works
- Design 'branch_search' produced 4 breakthroughs — double down here

## Dead ends — do NOT retry
- Design 'perturbation' has 20 experiments, 0 keeps — abandon this approach

## Gaps — unexplored
- 18 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 36 exp, 2 breakthroughs, rate 6%, best 1.13671782e-23
- agent1: 13 exp, 3 breakthroughs, rate 23%, best 1.13671782e-23
- agent2: 49 exp, 0 breakthroughs, rate 0%, best 8.31925483e-19
- agent3: 27 exp, 2 breakthroughs, rate 7%, best 1.17089535e-16

## Alerts
- crash_streak: Agent has 3 consecutive crashes — likely broken config or OOM
- crash_streak: Agent has 4 consecutive crashes — likely broken config or OOM
- crash_streak: Agent has 5 consecutive crashes — likely broken config or OOM

## Recent blackboard (last 20 entries)
**Interpretation:** The basin structure exhibits a **pitchfork bifurcation with hysteresis**:
1. Large |u_offset| drives to matching-sign branch (stable)
2. At critical u_offset ≈ ±0.57-0.62: bifurcation point where branch assignment inverts
3. At exact singularity (u_offset≈-0.62): Newton solver encounters singular Jacobian (crash)
4. The K(θ)=0.3cos(θ) term breaks symmetry, creating this intricate nonlinear structure
**Technical note:** The degraded residuals (7.70e-12 vs 3.25e-12) at boundary suggest solver is
struggling near bifurcation point, indicating ill-conditioning of the Newton system.
**Status:** Basin bifurcation topology fully mapped. Boundary transitions characterized to ±0.01 precision.
Ready for next direction: investigate K_frequency/K_amplitude parameter sweep to understand bifurcation genesis.
## AGENT3 FINAL SUMMARY (2026-04-06)
**Experiments: 21 contributed to 96-experiment campaign**
**Key results:**
- Negative basin boundary: u_offset ≈ -0.85 (fold point)
- Basin crossing: confirmed +0.57→negative, 0.595→trivial (hysteresis mechanism)
- Positive basin reaches negative offsets (asymmetry from K(θ))
- Ultra-precision trivial: 1.17e-16 at u_offset=0.595
**Research status:** Solution space fully characterized. Complex bifurcation structure documented.
Three-branch optimization complete (1.58e-15 trivial, 3.25e-12 ±1 branches), with trivial achieving
machine-zero at u_offset=0.1, n_mode=2, amplitude=0.3 (agent1 discovery: 1.137e-23).
**Next phase:** Investigate K(θ) parameter variations and bifurcation mechanisms.
