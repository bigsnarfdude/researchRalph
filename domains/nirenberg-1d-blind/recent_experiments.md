# Recent Experiments — nirenberg-1d-blind

**Best: 2.80498569e-21 (exp001)** | Total: 30 | Breakthroughs: 1 | Crashes: 10

### ↑ exp026 — 6.69306133e-20 (= best)
- **Agent:** agent0 | **Design:** branch_search | **Status:** discard
- **What:** u_offset=0.3 branch boundary sweep
- **Outcome:** INCREMENTAL

### ↑ exp027 — 3.05421974e-13 (= best)
- **Agent:** agent0 | **Design:** branch_search | **Status:** discard
- **What:** u_offset=0.5 branch boundary sweep
- **Outcome:** INCREMENTAL
- **Redundant with:** exp026

### ✗ exp028 — CRASH
- **Agent:** agent1 | **Design:** solver_param | **Status:** crash
- **What:** u_offset=0.9 fourier64 newton_tol=1e-14 maxiter=500 deep converge
- **Outcome:** CRASH
- **Redundant with:** exp022

### ↑ exp029 — 6.06963967e-15 (= best)
- **Agent:** agent0 | **Design:** branch_search | **Status:** discard
- **What:** u_offset=0.4 branch boundary bisect
- **Outcome:** INCREMENTAL
- **Redundant with:** exp026

### ↓ exp030 — 3.30235937e-13 (= best)
- **Agent:** agent0 | **Design:** branch_search | **Status:** discard
- **What:** u_offset=0.45 branch boundary bisect
- **Outcome:** REGRESSION
- **Redundant with:** exp026

## Score trajectory (all)
| exp | score | outcome |
|-----|-------|---------|
| exp001 | 2.80498569e-21 | ★ BREAKTHROUGH |
| exp002 | 2.88051949e-13 | ↓ REGRESSION |
| exp003 | 3.5336173e-13 | ↓ REGRESSION |
| exp004 | 2.80498569e-21 | → PLATEAU |
| exp005 | CRASH | ✗ CRASH |
| exp006 | CRASH | ✗ CRASH |
| exp007 | CRASH | ✗ CRASH |
| exp008 | 2.41595655e-09 | ↓ REGRESSION |
| exp009 | 5.72745859e-09 | ↓ REGRESSION |
| exp010 | 2.99966388e-13 | ↑ INCREMENTAL |
| exp011 | CRASH | ✗ CRASH |
| exp012 | CRASH | ✗ CRASH |
| exp013 | CRASH | ✗ CRASH |
| exp014 | CRASH | ✗ CRASH |
| exp015 | 8.81934685e-11 | ↓ REGRESSION |
| exp016 | 8.77448543e-11 | ↑ INCREMENTAL |
| exp017 | 2.70270424e-13 | ↑ INCREMENTAL |
| exp018 | CRASH | ✗ CRASH |
| exp019 | 3.55289665e-13 | ↑ INCREMENTAL |
| exp020 | 8.81934685e-11 | ↓ REGRESSION |
| exp021 | 3.3791155e-13 | ↑ INCREMENTAL |
| exp022 | 2.88051949e-13 | ↑ INCREMENTAL |
| exp023 | 8.81934684e-11 | ↓ REGRESSION |
| exp024 | CRASH | ✗ CRASH |
| exp025 | 1.2790499e-12 | ↓ REGRESSION |
| exp026 | 6.69306133e-20 | ↑ INCREMENTAL |
| exp027 | 3.05421974e-13 | ↑ INCREMENTAL |
| exp028 | CRASH | ✗ CRASH |
| exp029 | 6.06963967e-15 | ↑ INCREMENTAL |
| exp030 | 3.30235937e-13 | ↓ REGRESSION |
