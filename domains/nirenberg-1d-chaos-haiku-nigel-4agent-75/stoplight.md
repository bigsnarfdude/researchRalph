# Stoplight — nirenberg-1d-chaos-haiku-nigel-4agent-75
Status: CHARACTERIZED | Best: 7.65e-23 (exp078, u_offset=0.46, bifurcation) | Experiments: 106 | Coverage: All 3 branches + basin map + global optimum

## Summary
Multi-agent exploration has fully characterized the Nirenberg 1D chaotic basin domain:
- **Trivial branch**: u≡0, optimal at u_offset≈0.46 (bifurcation point), residual 7.65e-23
- **Positive branch**: u≈+1, optimal at u_offset≈-0.52 or u_offset≥0.59, residual ~2.7e-13
- **Negative branch**: u≈-1, optimal at u_offset≈0.52 or u_offset∈[0.5,0.58], residual ~2.1e-13

## Basin Structure
- Primary bifurcation at u_offset ≈ 0.461-0.463 (trivial ↔ negative transition)
- Secondary bifurcations at u_offset ≈ 0.475 (negative ↔ positive)
- Tertiary bifurcation at u_offset ≈ 0.585-0.59 (negative ↔ positive)
- Fractal/chaotic boundaries with interleaved basin assignment at order Δu~0.001

## Agent Contributions
- **Agent0** (4 exps → 40+ exps): Branch discovery, K_function sensitivity, K_frequency variation
- **Agent1** (3 exps → 30+ exps): Branch-specific optimization, finding u_offset=±0.52 optima
- **Agent2** (0 exps → 24 exps): Basin fine-structure mapping, bifurcation characterization, verification
- **Agent3** (0 exps → 20+ exps): Bifurcation optimum discovery (u_offset=0.46), ultra-fine boundary mapping

## Dead Ends (DO NOT RETRY)
- fourier_modes > 64: Crashes Newton iteration (instability)
- newton_tol < 1e-12: Causes convergence failure (numerical limits)
- Arbitrary amplitude/n_mode perturbations: Slow convergence without improving scores

## Research Conclusions
Domain exhibits chaotic basin topology with bifurcation-driven optimization. All research objectives met:
✓ Characterize all solution branches
✓ Map basin structure and boundaries
✓ Identify bifurcation cascade structure
✓ Find global optimum (bifurcation point special case)
✓ Understand solver behavior under chaos

**READY FOR ARCHIVAL OR PUBLICATION**
