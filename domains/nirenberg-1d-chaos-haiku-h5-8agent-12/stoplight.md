# Stoplight — nirenberg-1d-chaos-haiku-h5-8agent-12
Status: STAGNANT | Best: 0.0 (exp001) | Experiments: 14 | Stagnation: 13 since last breakthrough

## Dead ends — do NOT retry
- Design 'agent1' has 4 experiments, 0 keeps — abandon this approach
- Design 'agent2' has 3 experiments, 0 keeps — abandon this approach
- Design 'agent3' has 3 experiments, 0 keeps — abandon this approach

## Agents
- Negative branch (u_offset=-0.9, n=300, tol=1e-11): 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- Phase 1: Negative branch, scipy n=300 tol=1e-11, u_offset=-0.9: 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- Phase 1: Positive branch, scipy n=300 tol=1e-11, u_offset=0.9: 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- Phase 1: Trivial branch, scipy n=196 tol=1e-12, u_offset=0, amp=0: 1 exp, 0 breakthroughs, rate 0%, best 0.0
- Positive branch (u_offset=0.9, n=300, tol=1e-11): 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- Trivial branch baseline (u_offset=0, amp=0): 1 exp, 0 breakthroughs, rate 0%, best 0.0
- confirm trivial branch (u_offset=0): 1 exp, 1 breakthroughs, rate 0%, best 0.0
- negative branch (u_offset=-0.9): 1 exp, 0 breakthroughs, rate 0%, best 6.18636244e-10
- positive branch Fourier 1-mode newton_tol=1e-12 u_offset=0.9: 1 exp, 0 breakthroughs, rate 0%, best 5.55111512e-17
- positive branch with looser tol (u_offset=0.9, tol=1e-8): 1 exp, 0 breakthroughs, rate 0%, best 2.93396177e-10
- positive branch, u_offset=+0.9: 1 exp, 0 breakthroughs, rate 0%, best —
- positive branch, u_offset=+0.9, tol=1e-11: 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- positive branch, u_offset=0.9, scipy n=300 tol=1e-11: 1 exp, 0 breakthroughs, rate 0%, best 3.25175408e-12
- target positive branch (u_offset=0.9): 1 exp, 0 breakthroughs, rate 0%, best —

## Recent blackboard (last 20 entries)
CLAIM agent3: exp004 trivial scipy (n=300, tol=1e-12) → residual=0.0 mean=0.0 — branch=[trivial] **EXACT**
CLAIM agent3: exp010 positive scipy (n=300, tol=1e-11) → residual=3.25e-12 mean=+1.000 — branch=[positive]
CLAIM agent3: exp011 negative scipy (n=300, tol=1e-11) → residual=3.25e-12 mean=-1.000 — branch=[negative]
CLAIMED agent4: positive branch baseline with u_offset=+0.9
## agent2 Phase 1 — All three branches reproduced
CLAIM agent2: exp002 trivial scipy (n=196, tol=1e-12) → residual=0.0 mean=0.0 — branch=[trivial] **EXACT**
CLAIM agent2: exp005 positive scipy (n=300, tol=1e-11) → residual=3.25e-12 mean=+1.000 — branch=[positive]
CLAIM agent2: exp008 negative scipy (n=300, tol=1e-11) → residual=3.25e-12 mean=-1.000 — branch=[negative]
## agent1 observations
CLAIM agent1: exp001 trivial (u_offset=0, amp=0, tol=1e-12) → residual=0.0 mean=0.0 — branch=[trivial]
CLAIM agent1: exp006 positive (u_offset=0.9, amp=0.01, tol=1e-8) → residual=2.93e-10 mean=+1.000 — branch=[positive]
CLAIM agent1: exp009 negative (u_offset=-0.9, amp=0.01, tol=1e-8) → residual=6.19e-10 mean=-1.000 — branch=[negative]
**OBSERVATION:** agent2,3 getting 3.25e-12 on nontriv with n=300 tol=1e-11. Agent1 looser tol (1e-8) gives 1-2 OOM worse. Next: try matching agent2/3 tolerance.
