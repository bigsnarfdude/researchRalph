# Recent Experiments — sae-island-isl-a

**Best: 0.7019 (EXP-008)** | Total: 9 | Breakthroughs: 4 | Crashes: 0

### ★ EXP-005 — 0.4743 (Δ -0.2276 from best)
- **Agent:** agent0 | **Design:** jumprelu-v2-l0-1p0 | **Status:** keep
- **What:** JumpReLU calibration: l0_coefficient=1.0 (1000x EXP-002's 1e-3), full 200M samples. Goal: find coefficient landing L0 near baseline's k=25 before comparing F1 to 0.9894.
- **Outcome:** BREAKTHROUGH

### ↑ EXP-006 — 0.1534 (Δ -0.5485 from best)
- **Agent:** agent0 | **Design:** jumprelu-v3-l0-0p4 | **Status:** keep
- **What:** JumpReLU calibration round 3: l0_coefficient=0.4 (log-log interpolated from 1e-3->L0=2730 and 1.0->L0=12.9 to target L0~25), full 200M samples.
- **Outcome:** INCREMENTAL

### ↑ EXP-007 — 0.3733 (Δ -0.3286 from best)
- **Agent:** agent0 | **Design:** jumprelu-v4-l0-2p0 | **Status:** keep
- **What:** JumpReLU calibration round 4: l0_coefficient=2.0 (2x EXP-005). Testing whether F1 continues improving past L0=12.9 as coefficient rises further (F1 trend so far monotonically increasing with coefficient: 1e-3->0.033, 0.4->0.153, 1.0->0.474).
- **Outcome:** INCREMENTAL

### ★ EXP-008 — 0.7019 (= best)
- **Agent:** agent0 | **Design:** jumprelu-v5-l0-0p7 | **Status:** keep
- **What:** JumpReLU calibration round 5: l0_coefficient=0.7, refining around the peak found at coeff=1.0 (F1=0.4743, L0=12.9), bracketed by 0.4 (F1=0.153) and 2.0 (F1=0.373).
- **Outcome:** BREAKTHROUGH

### ↑ EXP-009 — 0.6493 (Δ -0.0526 from best)
- **Agent:** agent0 | **Design:** jumprelu-v6-l0-0p6 | **Status:** keep
- **What:** JumpReLU calibration round 6: l0_coefficient=0.6, tightening bracket around the 0.7 peak (F1=0.7019, L0=22.5, very close to baseline k=25).
- **Outcome:** INCREMENTAL

## Score trajectory (all)
| exp | score | outcome |
|-----|-------|---------|
| EXP-001 | 0.0468 | ★ BREAKTHROUGH |
| EXP-002 | 0.0331 | ↓ REGRESSION |
| EXP-003 | 0.0469 | ★ BREAKTHROUGH |
| EXP-004 | 0.006 | ↓ REGRESSION |
| EXP-005 | 0.4743 | ★ BREAKTHROUGH |
| EXP-006 | 0.1534 | ↑ INCREMENTAL |
| EXP-007 | 0.3733 | ↑ INCREMENTAL |
| EXP-008 | 0.7019 | ★ BREAKTHROUGH |
| EXP-009 | 0.6493 | ↑ INCREMENTAL |
