
## agent2 — Cycle 1 Learnings
- Fourier with u_offset=0.9, amp=0.1 crashes at 64 modes — need u_offset=1.0, amp=0.15 for convergence
- Fourier 32 modes gives slightly better residuals than 64 for non-trivial branches (~1.4e-13 vs ~2.6e-13)
- scipy with 196 nodes, tol=1e-10 gives ~9.4e-11 — 2 orders of magnitude worse than Fourier
- newton_tol=1e-14 crashes; solver plateaus around 1e-13 for non-trivial branches
- Trivial branch: Fourier gives exact 0.0 residual
