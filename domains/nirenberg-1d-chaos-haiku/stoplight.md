# Stoplight — nirenberg-1d-chaos-haiku
Status: STAGNANT | Best: 0.0 (exp001) | Experiments: 24 | Stagnation: 23 since last breakthrough

## Dead ends — do NOT retry
- Design 'agent0' has 12 experiments, 0 keeps — abandon this approach
- Design 'agent1' has 12 experiments, 0 keeps — abandon this approach

## Agents
- boundary basin: u_offset=0.5 (between trivial and positive): 1 exp, 0 breakthroughs, rate 0%, best 3.25175165e-12
- boundary test: u_offset=0.3: 1 exp, 0 breakthroughs, rate 0%, best 1.50179291e-14
- boundary test: u_offset=0.4: 1 exp, 0 breakthroughs, rate 0%, best 5.87544152e-20
- negative branch, n_nodes=196, tol=1e-11 (optimal config): 1 exp, 0 breakthroughs, rate 0%, best 1.47161968e-12
- negative branch, u_offset=-0.9, n_nodes=300, tol=1e-11: 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- negative branch: u_offset=-0.9, n_nodes=300, tol=1e-11: 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- positive branch matching agent0: u_offset=0.9, n_nodes=300, tol=1e-11: 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- positive branch with conservative u_offset=0.8: 1 exp, 0 breakthroughs, rate 0%, best —
- positive branch with mode-2 perturbation, amplitude=0.1: 1 exp, 0 breakthroughs, rate 0%, best 3.25175165e-12
- positive branch with n_nodes=280, u_offset=0.8: 1 exp, 0 breakthroughs, rate 0%, best —
- positive branch with n_nodes=300 (max), solver_tol=1e-10: 1 exp, 0 breakthroughs, rate 0%, best 2.60019368e-11
- positive branch with tighter tolerance 1e-12: 1 exp, 0 breakthroughs, rate 0%, best —
- positive branch, n_nodes=194, tol=1e-11: 1 exp, 0 breakthroughs, rate 0%, best 1.51627807e-12
- positive branch, n_nodes=195, tol=1e-11: 1 exp, 0 breakthroughs, rate 0%, best 1.51627807e-12
- positive branch, n_nodes=196, tol=1e-11 (match trivial nodes): 1 exp, 0 breakthroughs, rate 0%, best 1.47161968e-12
- positive branch, n_nodes=196, tol=1e-12 (push tol): 1 exp, 0 breakthroughs, rate 0%, best —
- positive branch, n_nodes=210, tol=1e-11 (fine-tune nodes): 1 exp, 0 breakthroughs, rate 0%, best 9.51700331e-12
- positive branch, n_nodes=220, tol=1e-11: 1 exp, 0 breakthroughs, rate 0%, best 9.51700331e-12
- positive branch, n_nodes=250, tol=1e-10 (looser tol test): 1 exp, 0 breakthroughs, rate 0%, best 4.50208716e-11
- positive branch, n_nodes=300, tol=5e-12 (between current and 1e-12): 1 exp, 0 breakthroughs, rate 0%, best 4.9993886e-12
- positive branch, u_offset=0.9, n_nodes=300, tol=1e-11: 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- target positive branch, u_offset=0.9: 1 exp, 0 breakthroughs, rate 0%, best —
- trivial branch baseline, u_offset=0: 1 exp, 1 breakthroughs, rate 0%, best 0.0
- trivial branch with tighter tolerance 1e-12: 1 exp, 0 breakthroughs, rate 0%, best 0.0

## Alerts
- deep_stagnation: No improvement in 18 experiments — search space may be exhausted or agents are stuck

## Recent blackboard (last 20 entries)
## Branch Coverage Status
- [x] Trivial (u≈0): exp001 residual=0.0, mean=0.0, n_nodes=196, tol=1e-12
- [x] Positive (u≈+1): exp002 residual=3.25e-12, mean=1.0, n_nodes=300, tol=1e-11
- [x] Negative (u≈-1): exp004 residual=3.25e-12, mean=-1.0, n_nodes=300, tol=1e-11
## Key Observations
- All three branches found in baseline (agent0 only, exp 1-4)
- Non-trivial residuals (3.25e-12) ~ 9x worse than prior Opus runs (2.83e-22)
- Symmetry: positive and negative branches show identical residuals
- exp003 missing (numbering jump suggests solver flag)
## Agent1 (Chaos) Status
- **exp003, 005, 006: ALL CRASHED** — scipy internal limit: "maximum number of mesh nodes exceeded"
- exp002_positive: Failed with same error (u_offset=0.9, amp=0.1)
- exp004_pos_conservative: Failed (u_offset=0.8, amp=0.05)
- **Pattern:** Agent1 attempting positive branch variations, all hitting solver resource limits
- **Hypothesis:** Chaos prompt is not causing crashes directly; scipy backend hitting internal limits on agent1's configs
## Experiment Log
