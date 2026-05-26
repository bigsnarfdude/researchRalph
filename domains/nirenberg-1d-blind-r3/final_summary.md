# Nirenberg 1D Blind Domain — Final Summary

## Experiments Completed
- **Total**: 171 experiments
- **Agents**: 8 (agent0–agent7)
- **Successful**: 156 discard + 5 keep = 161 valid runs
- **Crashes**: 15 (all attempted high-precision convergence)

## Solution Space Mapping

### Branch Discovery
All three solution branches identified and optimized:

| Branch | u_offset | Best Residual | Config | Experiment |
|--------|----------|---------------|--------|------------|
| **Trivial** | 0.0 | 1.1e-21 | n_nodes=392, tol=1e-11 | exp130 (agent0) |
| **Positive** | 0.9 | 1.46e-12 | n_nodes=392, tol=1e-11 | exp115 (agent0) |
| **Negative** | -0.9 | 1.46e-12 | n_nodes=392, tol=1e-11 | exp117 (agent0) |

### Key Discoveries

1. **Mesh Density Optimization** (agent0/agent3):
   - Non-trivial branches showed continuous improvement with n_nodes: 300→350→360→370→380→390→392
   - Sweet spot at n_nodes=392: residual=1.46e-12
   - Degradation beyond n_nodes≥395 (residual→1e-11)
   - Trivial branch benefits even more: 1.1e-21

2. **Bifurcation Structure** (agent1/agent5/agent6):
   - Non-monotonic branch behavior discovered
   - Bistability region around u_offset≈0.55
   - Boundary points identified at u_offset≈0.28, 0.45, 0.48, 0.55

3. **Convergence Characterization**:
   - Non-trivial branches hit stability floor near tol=1e-12 (crashes when attempted)
   - Trivial branch reaches machine precision (~1e-21)
   - Amplitude sensitivity: smaller amp (0.02) causes crashes; amp=0.05 stable

## Lessons Learned

### What Worked Well
✓ Systematic u_offset sweeps revealed all branches  
✓ Fine mesh optimization (100→392) yielded 2.2× improvement  
✓ Multi-agent exploration uncovered non-obvious bifurcation structure  

### What Didn't Work
✗ Tolerance tightening (tol < 1e-11) caused numerical instability  
✗ Extreme mesh sizes (n_nodes ≥ 400) introduced ill-conditioning  
✗ Phase/mode perturbations showed no benefit (converged to same residual)  

### Surprising Finding
The convergence "floor" at 3e-12 (early in exploration) was not fundamental—it was an artifact of insufficient mesh refinement. Only systematic sweeping (300→392) revealed the true optimum. This suggests **convergence floors require empirical stress-testing**, not theoretical assumptions.

## Current Frontier

**Still open**:
1. **Bifurcation diagram**: Full mapping of u_offset → (branch, residual) space
2. **Solution profiles**: Access to u(θ) to visualize convergence quality
3. **Tolerance boundary**: Exact limit between stability (1e-11) and crashes (1e-12)
4. **Alternative solvers**: Compare Fourier-Newton vs. scipy.integrate.solve_bvp

## Conclusion

The domain has been well-explored. Three branches found, optimized to near-machine-precision, and their convergence characteristics mapped. The optimal configuration (n_nodes=392, tol=1e-11, u_offset∈{0, ±0.9}) is stable and reproducible. Further gains likely require:
- Alternative solvers (to break 1.46e-12 floor on non-trivial branches)
- Solution visualization (to understand optimality of n_nodes=392)
- Bifurcation analysis (to exploit the multi-branch structure)
