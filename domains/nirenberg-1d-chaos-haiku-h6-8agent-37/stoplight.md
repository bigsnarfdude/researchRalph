# Stoplight — nirenberg-1d-chaos-haiku-h6-8agent-37
Status: STAGNANT | Best: 0.0 (exp001) | Experiments: 28 | Stagnation: 27 since last breakthrough

## Dead ends — do NOT retry
- Design 'agent0' has 10 experiments, 0 keeps — abandon this approach
- Design 'agent1' has 3 experiments, 0 keeps — abandon this approach
- Design 'agent2' has 5 experiments, 0 keeps — abandon this approach
- Design 'agent3' has 7 experiments, 0 keeps — abandon this approach
- Design 'agent4' has 3 experiments, 0 keeps — abandon this approach

## Agents
- agent2: negative branch hunt, u_offset=-0.9, n_nodes=300, tol=1e-11: 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- agent2: positive branch hunt, u_offset=0.9: 1 exp, 0 breakthroughs, rate 0%, best —
- agent2: positive branch refinement, u_offset=0.9, tol=1e-10: 1 exp, 0 breakthroughs, rate 0%, best 2.60019365e-11
- agent2: positive branch, u_offset=0.9, n_nodes=300, tol=1e-11: 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- agent2: trivial branch baseline, u_offset=0: 1 exp, 0 breakthroughs, rate 0%, best 0.0
- negative branch high precision u_offset=-0.9: 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- negative branch, scipy n=300 tol=1e-11, u_offset=-0.9: 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- negative branch, u_offset=-0.9 amp=0.1: 1 exp, 0 breakthroughs, rate 0%, best —
- positive branch conservative, u_offset=0.5 amp=0.1: 1 exp, 0 breakthroughs, rate 0%, best 2.20823255e-15
- positive branch high precision u_offset=0.6: 1 exp, 0 breakthroughs, rate 0%, best 3.25175165e-12
- positive branch test, u_offset=0.9: 1 exp, 0 breakthroughs, rate 0%, best 0.0
- positive branch, scipy n=300 tol=1e-11, u_offset=0.9: 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- positive branch, u_offset=0.9: 1 exp, 0 breakthroughs, rate 0%, best —
- test bifurcation region at u_offset=0.55 with tol=1e-10: 1 exp, 0 breakthroughs, rate 0%, best 9.37325052e-11
- trivial branch test, u_offset=0.0: 1 exp, 0 breakthroughs, rate 0%, best 0.0
- trivial branch, scipy n=196 tol=1e-12, u_offset=0: 1 exp, 0 breakthroughs, rate 0%, best 0.0
- u_offset=-0.7, hunt negative branch: 1 exp, 0 breakthroughs, rate 0%, best 2.60019365e-11
- u_offset=0, test trivial branch: 1 exp, 1 breakthroughs, rate 0%, best 0.0
- u_offset=0.1, gentle perturbation from trivial: 1 exp, 0 breakthroughs, rate 0%, best 9.15022415e-21
- u_offset=0.5, smaller positive offset: 1 exp, 0 breakthroughs, rate 0%, best —
- u_offset=0.52, tol=1e-9, bifurcation boundary: 1 exp, 0 breakthroughs, rate 0%, best 1.58906086e-13
- u_offset=0.55, tol=1e-9, find minimum positive offset: 1 exp, 0 breakthroughs, rate 0%, best 7.01894433e-10
- u_offset=0.575, tol=1e-10: 1 exp, 0 breakthroughs, rate 0%, best 4.43916024e-13
- u_offset=0.6, tol=1e-9, boundary search: 1 exp, 0 breakthroughs, rate 0%, best 2.07993383e-10
- u_offset=0.60, tol=1e-10: 1 exp, 0 breakthroughs, rate 0%, best 9.37325053e-11
- u_offset=0.7, tol=1e-10, refine positive: 1 exp, 0 breakthroughs, rate 0%, best 2.60019368e-11
- u_offset=0.7, tol=1e-8, relax solver: 1 exp, 0 breakthroughs, rate 0%, best 2.07993308e-10
- u_offset=0.9, hunt positive branch: 1 exp, 0 breakthroughs, rate 0%, best —

## Alerts
- deep_stagnation: No improvement in 22 experiments — search space may be exhausted or agents are stuck

## Recent blackboard (last 20 entries)
## Branch Discovery (agent0, Exp 1-7)
CLAIM agent0: residual=0.0 mean=0.0 norm=0.0 (exp001) — branch=TRIVIAL | u_offset=0.0 | solver_tol=1e-12
CLAIM agent0: residual=2.6e-11 mean=+1.0 norm=1.003 (exp007) — branch=POSITIVE | u_offset=0.7 | solver_tol=1e-10
CLAIM agent0: residual=2.6e-11 mean=-1.0 norm=1.003 (exp009) — branch=NEGATIVE | u_offset=-0.7 | solver_tol=1e-10
## Key Finding: Solver Robustness
- u_offset < 0.5: converges to trivial branch
- 0.5 < u_offset < 0.7: crashes with tight tolerance (1e-12)
- u_offset ≈ 0.7 with tol ≥ 1e-10: stable convergence to positive branch
- Symmetry preserved: negative branch mirrors positive at u_offset=-0.7
## agent2 Branch Confirmation (Exp 8, 15, 19)
CLAIM agent2: residual=0.0 mean=0.0 norm=0.0 (exp008) — branch=TRIVIAL | u_offset=0.0 | n_nodes=196, tol=1e-12
CLAIM agent2: residual=3.25e-12 mean=+1.0 norm=1.0025 (exp015) — branch=POSITIVE | u_offset=0.9 | n_nodes=300, tol=1e-11
CLAIM agent2: residual=3.25e-12 mean=-1.0 norm=1.0025 (exp019) — branch=NEGATIVE | u_offset=-0.9 | n_nodes=300, tol=1e-11
**agent2 observation:** Both non-trivial branches achieve same residual at higher u_offset (0.9 vs 0.7 from agent0) with n_nodes=300, tol=1e-11. Calibration.md reports scipy can reach 1e-22 to 1e-27 for non-trivial branches. Testing tighter tolerance sweep next.
