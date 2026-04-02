# Recent Experiments — nirenberg-1d-blind

**Best: nan (exp001)** | Total: 9 | Breakthroughs: 1 | Crashes: 4

### ↑ exp005 — nan (= best)
- **Agent:** agent1 | **Design:** initial_cond | **Status:** discard
- **What:** u_offset=-0.9 negative branch search
- **Outcome:** INCREMENTAL
- **Redundant with:** exp004

### ✗ exp006 — nan (= best)
- **Agent:** agent0 | **Design:** solver_param | **Status:** crash
- **What:** u_offset=0.9 fourier_modes=128 positive branch
- **Outcome:** CRASH

### ✗ exp007 — nan (= best)
- **Agent:** agent1 | **Design:** solver_param | **Status:** crash
- **What:** u_offset=0.9 fourier_modes=128 newton_tol=1e-14 optimize positive branch
- **Outcome:** CRASH

### ✗ exp008 — nan (= best)
- **Agent:** agent0 | **Design:** solver_param | **Status:** crash
- **What:** u_offset=1.0 fourier_modes=128 maxiter=200
- **Outcome:** CRASH

### ✗ exp009 — nan (= best)
- **Agent:** agent1 | **Design:** solver_param | **Status:** crash
- **What:** u_offset=0.9 fourier=128 tol=1e-14 maxiter=200
- **Outcome:** CRASH

## Score trajectory (all)
| exp | score | outcome |
|-----|-------|---------|
| exp001 | nan | ★ BREAKTHROUGH |
| exp002 | nan | ↑ INCREMENTAL |
| exp003 | nan | ↑ INCREMENTAL |
| exp004 | nan | ↑ INCREMENTAL |
| exp005 | nan | ↑ INCREMENTAL |
| exp006 | nan | ✗ CRASH |
| exp007 | nan | ✗ CRASH |
| exp008 | nan | ✗ CRASH |
| exp009 | nan | ✗ CRASH |
