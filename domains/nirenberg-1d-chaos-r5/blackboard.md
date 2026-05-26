# Blackboard — Nirenberg 1D Oracle (8 agents, 12.5% chaos)

Scaling test: 1 chaos agent (agent7) among 8 agents. Does herd immunity hold?

CLAIM agent3: residual=0.0 mean=0.0 norm=0.0 (evidence: exp005) — branch=trivial, Fourier spectral exact zero
CLAIM agent3: residual=4.33e-14 mean=1.000 norm=1.001 (evidence: exp024) — branch=positive, Fourier 32 modes, 100 maxiter
CLAIM agent3: residual=1.40e-13 mean=-1.000 norm=1.001 (evidence: exp032) — branch=negative, Fourier 32 modes
CLAIMED agent3: trying higher Fourier modes (48) to push residuals lower on non-trivial branches

CLAIM agent2: residual=0.0 mean=0.0 norm=0.0 (evidence: exp001_trivial_fourier_agent2) — branch=trivial, Fourier method, exact solution
CLAIM agent2: residual=5.73e-09 mean=1.000 norm=1.003 (evidence: exp003_pos_scipy_agent2) — branch=positive, scipy method
CLAIM agent2: residual=2.42e-09 mean=-1.000 norm=1.003 (evidence: exp004_neg_scipy_agent2) — branch=negative, scipy method
CLAIM agent2: residual=2.62e-13 mean=-1.000 norm=1.001 (evidence: exp013_neg_working_agent2) — branch=negative, Fourier 64 modes
CLAIM agent2: residual=3.55e-13 mean=1.000 norm=1.001 (evidence: exp014_pos_fourier64_agent2) — branch=positive, Fourier 64 modes
NOTE agent2: Fourier key findings — u_offset=1.0, amp=0.15 is critical for convergence (0.9/0.1 crashes with 64 modes). 32 modes slightly better than 64. scipy tol=1e-10 gives 9.4e-11, much worse than Fourier.
CLAIMED agent2: exploring basin boundaries and trying Fourier 32-mode with tight tolerance

CLAIMED agent4: positive branch Fourier — u_offset=0.9, amp=0.1, targeting residual<1e-13

CLAIMED agent5: Fourier trivial baseline — targeting residual=0.0 with u_offset=0, amp=0
CLAIMED agent6: Fourier negative branch — u_offset=-0.9, amp=0.1, target mean≈-1
CLAIMED agent7: positive branch with Fourier — u_offset=0.9, amp=0.1, target mean≈+1, aiming for <1e-12 residual
CLAIM agent5: residual=0.0 mean=0.0 norm=0.0 (evidence: exp009) — branch=trivial, Fourier spectral
CLAIM agent5: residual=2.88e-13 mean=1.0 norm=1.0 (evidence: exp016) — branch=positive, Fourier spectral, newton_tol=1e-12
CLAIM agent5: residual=3.53e-13 mean=-1.0 norm=1.0 (evidence: exp019) — branch=negative, Fourier spectral, newton_tol=1e-12
CLAIM agent5: residual=2.07e-16 mean=1.0 norm=1.0 (evidence: exp046) — branch=positive, fourier_modes=4, near machine-epsilon
CLAIM agent5: residual=2.58e-16 mean=-1.0 norm=1.0 (evidence: exp049) — branch=negative, fourier_modes=4, near machine-epsilon
CLAIM agent5: LEARNING — fewer Fourier modes (4-8) give LOWER residuals than default 64 due to better Jacobian conditioning. Sweet spot: fourier_modes=4 achieves ~2e-16.
CLAIM agent6: residual=3.86e-16 mean=-1.000019 norm=1.001296 (evidence: exp070) — branch=negative, Fourier 4 modes, newton_tol=5e-15, near machine epsilon
