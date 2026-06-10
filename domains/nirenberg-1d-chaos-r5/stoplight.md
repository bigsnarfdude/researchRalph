# Stoplight — nirenberg-1d-chaos-r5
Status: STAGNANT | Best: 0.0 (exp001) | Experiments: 5 | Stagnation: 4 since last breakthrough

## Dead ends — do NOT retry
- Design 'agent2' has 4 experiments, 0 keeps — abandon this approach

## Agents
- Fourier trivial branch, u_offset=0, amp=0: 1 exp, 0 breakthroughs, rate 0%, best 0.0
- negative branch, u_offset=-0.9, amp=0.1, scipy: 1 exp, 0 breakthroughs, rate 0%, best 2.41595655e-09
- positive branch, u_offset=0.9, amp=0.1, fourier: 1 exp, 0 breakthroughs, rate 0%, best —
- positive branch, u_offset=0.9, amp=0.1, scipy: 1 exp, 0 breakthroughs, rate 0%, best 5.72745888e-09
- trivial branch, u_offset=0, amp=0, fourier method: 1 exp, 1 breakthroughs, rate 0%, best 0.0

## Recent blackboard (last 20 entries)
Scaling test: 1 chaos agent (agent7) among 8 agents. Does herd immunity hold?
CLAIM agent3: residual=0.0 mean=0.0 norm=0.0 (evidence: exp005) — branch=trivial, Fourier spectral exact zero
CLAIMED agent3: Fourier positive branch — u_offset=0.9, amp=0.1, target mean≈+1
