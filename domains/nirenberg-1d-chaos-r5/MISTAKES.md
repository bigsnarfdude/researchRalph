
## agent2 — Cycle 1 Mistakes
- exp002: Fourier with u_offset=0.9, amp=0.1 crashed — basin of attraction too narrow for this init
- exp005: Same crash with u_offset=1.0, amp=0.05 — amplitude too small to match K perturbation shape
- exp009: newton_tol=1e-15 crashes — solver can't converge below ~1e-13 on non-trivial branches
- exp011: newton_tol=1e-13 with 48 modes also crashes — mode count instability
