# Recent Experiments — nirenberg-1d-chaos-r5

**Best: 0.0 (exp001)** | Total: 5 | Breakthroughs: 1 | Crashes: 1

### ★ exp001 — 0.0 (= best)
- **Agent:** trivial branch, u_offset=0, amp=0, fourier method | **Design:** agent2 | **Status:** 0.000000
- **What:** keep
- **Outcome:** BREAKTHROUGH

### ✗ exp002 — CRASH
- **Agent:** positive branch, u_offset=0.9, amp=0.1, fourier | **Design:** agent2 | **Status:** 0
- **What:** crash
- **Outcome:** CRASH

### → exp003 — 5.72745888e-09 (= best)
- **Agent:** positive branch, u_offset=0.9, amp=0.1, scipy | **Design:** agent2 | **Status:** 1.000218
- **What:** keep
- **Outcome:** PLATEAU
- **Redundant with:** exp001

### → exp004 — 2.41595655e-09 (= best)
- **Agent:** negative branch, u_offset=-0.9, amp=0.1, scipy | **Design:** agent2 | **Status:** -1.000218
- **What:** keep
- **Outcome:** PLATEAU
- **Redundant with:** exp001

### → exp005 — 0.0 (= best)
- **Agent:** Fourier trivial branch, u_offset=0, amp=0 | **Design:** agent3 | **Status:** 0.000000
- **What:** keep
- **Outcome:** PLATEAU
- **Redundant with:** exp001
