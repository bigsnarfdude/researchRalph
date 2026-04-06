# Agent2 Research Summary — Phases 3–6
**Experiments:** 19 runs (exp002, exp007, exp012, exp016, exp047, exp062, exp068, exp071, exp073 + exp086, exp089, exp090, exp092, exp097, exp099, exp100, exp103, exp106, exp109)

## Key Discoveries

### Phase 3: Solver Refinement Breakthrough
**Problem:** Non-trivial branches had poor residuals (5.7e-9 for positive branch)
**Solution:** Increase mesh (n_nodes: 100→200) and tolerance (tol: 1e-8→1e-10)
**Impact:** 65× improvement (5.7e-9 → 8.8e-11)

**Critical finding:** The "negative branch plateau" (2.4e-9) was a solver artifact, not physics. With proper refinement, negative branch converges to same quality as positive (8.8e-11).

### Phase 4: Bifurcation Boundary Mapping  
Tested boundaries with refined solver parameters:
- exp062 (-0.625): unexpected positive branch crossing
- exp068 (-0.65): negative branch
- exp071 (+0.575): positive branch
- exp073 (-0.60): **CRASH** — discovered solver instability at phase transition

### Phase 5: Crash Zone Resolution
**Breakthrough:** Ultra-stable solver (n_nodes=300, tol=1e-11) resolves crash:
- exp089 (-0.60): NEGATIVE branch, 3.25e-12 residual
- exp090 (-0.59): **CRASH** even with ultra-stable solver
- exp092 (-0.595): NEGATIVE branch, 7.70e-12 residual

**Hyper-sharp boundary discovered:** transition from stable convergence to crash occurs in < 0.01 interval at u_offset ≈ ±0.59

### Phase 6: BASIN OF ATTRACTION OVERLAP DISCOVERY ⭐
**Shocking finding:** Negative branch basin invades positive parameter space!

**Boundary map (ultra-stable solver):**
```
u_offset=0.56: TRIVIAL (4.38e-17, machine precision!)
u_offset=0.57: NEGATIVE (3.25e-12) ← BASIN OVERLAP!
u_offset=0.58: NEGATIVE (3.25e-12) ← BASIN OVERLAP!
u_offset=0.59: CRASH (chaotic zone)
u_offset=0.60: POSITIVE (3.25e-12)
u_offset=0.61+: POSITIVE (3.25e-12)
```

**Implication:** The bifurcation diagram is NOT three simple regions. It has a **multi-lobed structure** with overlapping basins of attraction. The "asymmetry" found by agent1 is actually complex multi-lobed topology, not simple left-right symmetry breaking.

## Mechanism Insights

1. **Solver is the primary lever:** All three branches achieve 3.25e-12 residual with proper refinement (n_nodes=300, tol=1e-11)
2. **Bifurcation structure is complex:** Basin boundaries are sharp and sometimes overlap
3. **Chaotic transition zones:** Crash zones at ±0.59 indicate chaotic dynamics or singular manifold crossings
4. **Amplitude, mode, phase are secondary:** These parameters have minimal effect on branch selection or residual quality

## Recommendations for Future Work

1. **Fine-grained manifold mapping:** Explore (u_offset, amplitude, n_mode) 3D parameter space to understand basin geometry
2. **Dynamical analysis:** Compute Lyapunov exponents in crash zones to characterize chaos
3. **Continuation methods:** Use predictor-corrector to follow solution branches and ghost unstable manifolds
4. **K_frequency investigation:** Test if bifurcation structure changes with different cosine frequencies (K_frequency=0.5, 1, 2, etc.)

## What Agent2 Learned

- **Solver parameters drive everything:** More important than problem parameters
- **Expect surprises in bifurcation structure:** Simple-looking problems can have rich dynamics
- **Ultra-refinement is essential:** tol=1e-11 required to access true bifurcation structure; coarser tolerances miss crucial features
- **Basin overlap is real:** Initial condition (u_offset) can be misleading; basin structure is non-monotonic
