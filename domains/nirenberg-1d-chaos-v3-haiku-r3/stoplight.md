# Stoplight — nirenberg-1d-chaos-v3-haiku-r3
Status: PLATEAU | Best: 0.0 (exp063) | Experiments: 117 | Stagnation: 54 since last breakthrough

## What works
- Design 'initial_cond' produced 4 breakthroughs — double down here

## Dead ends — do NOT retry
- Design 'branch_search' has 45 experiments, 0 keeps — abandon this approach
- Design 'perturbation' has 10 experiments, 0 keeps — abandon this approach

## Gaps — unexplored
- 34 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- agent0: 26 exp, 0 breakthroughs, rate 0%, best 0.0
- agent1: 29 exp, 5 breakthroughs, rate 17%, best 0.0
- agent2: 19 exp, 0 breakthroughs, rate 0%, best 4.37799025e-17
- agent3: 43 exp, 0 breakthroughs, rate 0%, best 5.69266279e-19

## Alerts
- deep_stagnation: No improvement in 45 experiments — search space may be exhausted or agents are stuck

## Recent blackboard (last 20 entries)
**Key Discoveries:**
- **Symmetry Validated:** ±1.0 branches achieve identical residuals when using safe u_offset values (±0.63 vs ±0.65)
- **Mesh Optimization:** Non-monotonic convergence—optimal at n_nodes=390 (positive), n_nodes=350 (negative); beyond 390 → degradation
- **Basin Overlap:** Negative basin invades positive parameter space (u_offset=0.57-0.58)
- **Crash Zones:** Sharp chaotic boundaries at ±0.59 with tol≤1e-11
- **Solver Limits:** tol=1e-12 crashes; tol=1e-11 is practical optimum
**Recommended Baseline for Future Agents:**
```yaml
u_offset: 0.58
n_nodes: 390
solver_tol: 1e-11
u_offset: -0.63
n_nodes: 350
solver_tol: 1e-11
u_offset: -0.62
n_nodes: 100-200
solver_tol: 1e-10
```
**Bifurcation Structure Summary:**
The problem exhibits a complex, multi-lobed bifurcation diagram with overlapping basins of attraction. The apparent "asymmetry" between positive and negative branches is not fundamental—both reach identical residuals—but rather arises from the intricate topology of the solution manifold and the K(θ) forcing structure.
