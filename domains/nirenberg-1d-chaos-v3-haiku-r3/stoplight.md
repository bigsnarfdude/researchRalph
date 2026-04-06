# Stoplight — nirenberg-1d-chaos-v3-haiku-r3
Status: PLATEAU | Best: 0.0 (exp063) | Experiments: 82 | Stagnation: 19 since last breakthrough

## What works
- Design 'initial_cond' produced 4 breakthroughs — double down here

## Dead ends — do NOT retry
- Design 'branch_search' has 22 experiments, 0 keeps — abandon this approach
- Design 'perturbation' has 5 experiments, 0 keeps — abandon this approach

## Gaps — unexplored
- 18 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 17 exp, 0 breakthroughs, rate 0%, best 0.0
- agent1: 29 exp, 5 breakthroughs, rate 17%, best 0.0
- agent2: 9 exp, 0 breakthroughs, rate 0%, best 2.61333321e-11
- agent3: 27 exp, 0 breakthroughs, rate 0%, best 5.69266279e-19

## Alerts
- deep_stagnation: No improvement in 15 experiments — search space may be exhausted or agents are stuck

## Recent blackboard (last 20 entries)
   - Negative: residual=7.70e-12 (exp067) ✓ (312× improvement vs baseline)
ASYMMETRY RESOLVED:
At ultra-precision (n_nodes=300, tol=1e-11), negative branch residual (7.70e-12) > positive branch residual (3.25e-12). **Asymmetry is REAL**, not solver artifact. Negative branch is mathematically harder to solve (2.37× worse). Likely due to nonlinearity structure: u³ term couples differently to negative perturbations.
HYPOTHESIS: Negative branch requires slightly finer mesh or asymmetric solver tuning due to the sign of K(θ) modulation on negative solution profile.
## Agent2 — Phase 4: Boundary Refinement & Stability Testing
CLAIMED agent2: Validate bifurcation boundaries with refined solvers (n_nodes=200, tol=1e-10) and test stability at phase transitions.
BOUNDARY MAPPING (5 experiments):
- exp062: u_offset=-0.625 → POSITIVE branch, residual=2.61e-11 (boundary crossing!)
- exp068: u_offset=-0.65 → NEGATIVE branch, residual=8.82e-11 ✓
- exp071: u_offset=+0.575 → POSITIVE branch, residual=2.61e-11 ✓
- exp073: u_offset=-0.60 → **CRASH** (solver divergence at phase transition)
**Critical finding: Solver instability at bifurcation.**
- Negative boundary with refined tol=1e-10: somewhere between -0.65 and -0.625
- u_offset=-0.60 is a crash zone (possibly chaotic region or near singular point)
- This aligns with agent3's finding that tol=1e-12 crashes—we're hitting the edge of numerical stability
- Agent3's success at tol=1e-11 suggests this is the practical limit for this problem
**Implication for next work:**
- Ultra-refinement (agent3's n_nodes=300, tol=1e-11) successfully avoids crash zones
- Boundary location at n_nodes=200 is unreliable near crash points
- u_offset=-0.60 region warrants investigation with stable solver only (n_nodes=300+, tol=1e-11+)
