# Stoplight — nirenberg-1d-blind-r1
Status: ACTIVE | Best: 0.0 (exp040) | Experiments: 48 | Stagnation: 8 since last breakthrough

## What works
- Design 'solver_param' produced 4 breakthroughs — double down here

## Dead ends — do NOT retry
- Design 'perturbation' has 7 experiments, 0 keeps — abandon this approach

## Agents
- agent0: 17 exp, 3 breakthroughs, rate 18%, best 0.0
- agent1: 14 exp, 0 breakthroughs, rate 0%, best 7.78244459e-12
- agent2: 11 exp, 1 breakthroughs, rate 9%, best 3.25175165e-12
- agent3: 6 exp, 1 breakthroughs, rate 17%, best 3.25175165e-12

## Recent blackboard (last 20 entries)
CLAIM agent0: exp001 trivial branch — residual=5.64e-11 mean=0.0 norm=0.0 (u_offset=0)
CLAIM agent0: exp002 positive branch — residual=5.73e-09 mean=+1.0 norm=1.0 (u_offset=+0.9)
CLAIM agent0: exp003 negative branch — residual=2.42e-09 mean=-1.0 norm=1.0 (u_offset=-0.9)
**Status:** All three branches discoverable with baselines. Trivial branch achieves best residual (5.64e-11).
Exploring: Can we improve positive/negative branches by tuning amplitude, n_mode, or solver_tol?
## Optimization — agent0 (cycle 2)
**SOLVER_TOL BREAKTHROUGH:** Systematic exploration shows solver_tol=1e-10 + fine mesh (n_nodes=300) unlocks machine precision.
CLAIM agent0: exp021 trivial + solver_tol + mesh — residual=2.98e-13 mean=0.0 norm=0.0 (n_nodes=300, solver_tol=1e-10) ⭐ BREAKTHROUGH
CLAIM agent0: exp024 positive + solver_tol + mesh — residual=2.60e-11 mean=+1.0 norm=1.003 (n_nodes=300, solver_tol=1e-10)
CLAIM agent0: exp026 negative + solver_tol + mesh — residual=8.59e-11 mean=-1.0 norm=1.003 (n_nodes=300, solver_tol=1e-10)
**Key findings:**
- Trivial branch: 5.64e-11 → 2.98e-13 (50× improvement via mesh+tol)
- Positive branch: 5.73e-09 → 2.60e-11 (220× improvement)
- Negative branch: 2.42e-09 → 8.59e-11 (28× improvement)
- solver_tol=1e-12 causes crashes (unstable)
- solver_tol=1e-10 is sweet spot
- n_nodes plateau at 300 (400 no improvement)
- Phase/amplitude variations don't improve beyond this configuration
- Branch asymmetry: positive < negative (opposite of initial assumptions)
**Status:** Exploring residual limit. Next: check solution smoothness, consider alternate solvers.
