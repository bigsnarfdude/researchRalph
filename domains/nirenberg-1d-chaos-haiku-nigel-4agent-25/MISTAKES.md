# Agent Mistakes & Failures

## exp026: fourier_modes=128 with u_offset=0.9 → CRASH
- **What:** Tested increasing Fourier modes from 64 to 128 on positive branch
- **Result:** Solver crash (residual reported as "crash", nan solution)
- **Lesson:** Higher mode count may destabilize Newton convergence for non-trivial branches; 64 modes appears optimal. Do not increase without tuning newton_tol.

## exp034: newton_tol=1e-15 with u_offset=0.9 → CRASH
- **What:** Tightened Newton tolerance from 1e-12 to 1e-15 on positive branch
- **Result:** Solver crash (residual="crash", nan solution)
- **Lesson:** Current config (newton_tol=1e-12) is at the precision limit; tighter tolerance causes numerical instability. Already at machine precision residuals (~2.67e-13).

## exp047: newton_tol=1e-14 on chaotic region (agent3)
- **What:** Attempted to tighten tolerance on chaotic u_offset=0.48 region to test solver sensitivity
- **Result:** Crash after 2 seconds
- **Lesson:** Chaotic region is sensitive; even modest tolerance tightening destabilizes solver. Keep newton_tol at 1e-12.
