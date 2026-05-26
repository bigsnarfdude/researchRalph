# Stoplight — nirenberg-1d-chaos-haiku-nigel-h1
Status: STAGNANT | Best: 0.0 (exp003) | Experiments: 21 | Stagnation: 18 since last breakthrough

## Dead ends — do NOT retry
- Design 'agent1' has 13 experiments, 0 keeps — abandon this approach
- Design 'agent0' has 8 experiments, 0 keeps — abandon this approach

## Gaps — unexplored
- 4 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- baseline trivial branch u_offset=0.0: 1 exp, 1 breakthroughs, rate 0%, best 0.0
- baseline trivial branch u_offset=0.46: 1 exp, 1 breakthroughs, rate 0%, best 7.64566347e-23
- boundary between trivial and positive u_offset=0.5: 1 exp, 0 breakthroughs, rate 0%, best 2.28944733e-13
- explore transition region u_offset=0.6: 1 exp, 0 breakthroughs, rate 0%, best 2.76001344e-13
- explore u_offset=1.2 beyond nominal range: 1 exp, 0 breakthroughs, rate 0%, best 3.42609616e-13
- negative branch u_offset=-0.9: 2 exp, 0 breakthroughs, rate 0%, best 2.66626099e-13
- negative branch with mode-3 perturbation amplitude=0.2: 1 exp, 0 breakthroughs, rate 0%, best 2.43563245e-13
- positive branch u_offset=0.9: 3 exp, 0 breakthroughs, rate 0%, best 7.64566347e-23
- positive branch with amplitude=0.2 perturbation: 1 exp, 0 breakthroughs, rate 0%, best 3.79436503e-13
- positive branch with mode-2 perturbation amplitude=0.1: 1 exp, 0 breakthroughs, rate 0%, best 2.51368744e-13
- positive branch, higher Fourier modes (128): 1 exp, 0 breakthroughs, rate 0%, best —
- positive branch, more Newton iterations (200): 1 exp, 0 breakthroughs, rate 0%, best 2.66626099e-13
- positive branch, tighter Newton tolerance (1e-14): 1 exp, 0 breakthroughs, rate 0%, best —
- test u_offset=0.4 boundary: 1 exp, 0 breakthroughs, rate 0%, best 4.40113707e-21
- u_offset=-0.5 basin boundary: 1 exp, 0 breakthroughs, rate 0%, best 2.28944733e-13
- u_offset=0.55 basin boundary: 1 exp, 0 breakthroughs, rate 0%, best 2.13371637e-13
- u_offset=0.6 basin boundary: 1 exp, 0 breakthroughs, rate 0%, best 2.76001344e-13
- u_offset=0.7 boundary exploration: 1 exp, 0 breakthroughs, rate 0%, best 3.48477681e-13

## Alerts
- deep_stagnation: No improvement in 16 experiments — search space may be exhausted or agents are stuck

## Recent blackboard (last 20 entries)
## Claims
CLAIM agent1: Branch coverage complete — trivial (exp001, mean=0.0, res=7.64e-23), positive (exp004, mean=1.0, res=2.67e-13), negative (exp005, mean=-1.0, res=2.67e-13). All three solution branches mapped.
CLAIM agent0: Confirmed all three branches. exp003 trivial (mean=0.0, res=0.0), exp006 positive (mean=1.0, res=2.67e-13), exp007 negative (mean=-1.0, res=2.67e-13). Branch coverage complete.
## Responses
## Requests
