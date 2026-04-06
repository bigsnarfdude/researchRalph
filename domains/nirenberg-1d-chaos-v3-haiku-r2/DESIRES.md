## Desires Log — agent2

- Ability to visualize the three-branch bifurcation diagram (u_offset vs residual vs branch)
- Grid sweep results: test amplitude, mode, phase effects on each branch
- Perturbation stability: how do small oscillations affect convergence?

## agent1 Session Desires

- [ ] K-parameter sweep harness (K_amplitude ∈ [0, 1]) to find bifurcation set
- [ ] Spectral solver (e.g., Chebyshev) to compare convergence vs BVP
- [ ] Solution Fourier analysis tool (extract mode amplitudes, energy budget)
- [ ] Adaptive mesh refinement to see if residual floor can be beaten
- [ ] Direct energy minimization (instead of residual minimization) — does it find different solutions?
- [ ] Stability analysis of branches (eigenvalues of linearized operator)
