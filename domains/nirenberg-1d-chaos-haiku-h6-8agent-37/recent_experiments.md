# Recent Experiments — nirenberg-1d-chaos-haiku-h6-8agent-37

**Last Updated:** Apr 3 2026 (Session: agent1 basin mapping breakthrough)

## Summary of Recent Sessions

### Agent1 Phase 1-4: Complete Basin Discovery via Multi-Solver Validation (Exp 5, 10, 26, 100-220+)

**Major Breakthrough:** Scipy solver showed "chaos" (alternating basins in u_offset ∈ [0.52, 0.60]). Fourier spectral solver revealed it was numerical noise—all points in that region cleanly converge to negative branch with 5.55e-17 residual.

**Impact:** Revealed TRUE basin structure is non-monotone with isolated pockets (positive at u_offset=-0.5, negative at u_offset=[0.45, 0.60]).

## Solver Comparison (Critical Finding)

| Solver | u_offset=0.9 (Positive) | u_offset=-0.9 (Negative) | u_offset=0.576 (Chaos Region) |
|--------|---|---|---|
| **Scipy** (n=300, tol=1e-11) | residual=3.25e-12, mean=+1.0 | residual=3.25e-12, mean=-1.0 | ALTERNATES trivial/pos/neg |
| **Fourier 1-mode** (newton_tol=1e-12) | residual=5.55e-17, mean=+1.0 | residual=5.55e-17, mean=-1.0 | residual=5.55e-17, mean=-1.0 |

**Implication:** The 100,000× tighter residual floor (5.55e-17 vs 3.25e-12) resolves true basin structure.

## Basin Map (Fourier 1-Mode Ground Truth)

```
u_offset ≤ -0.9    : NEGATIVE (residual=5.55e-17)
u_offset = -0.5    : POSITIVE (residual=5.55e-17) ← ISOLATED!
u_offset ∈ [-0.48, -0.30]: TRIVIAL (residual=1e-15)
u_offset ∈ [0, 0.45]: TRIVIAL (residual=0, exact)
u_offset ∈ [0.45, 0.60]: NEGATIVE (residual=5.55e-17) ← INTERMEDIATE!
u_offset ∈ [0.62, 1.5]: POSITIVE (residual=5.55e-17)
```

**Key observation:** Not monotone! Negative basin has TWO disconnected components.

## Phase Summary

| Phase | Focus | Experiments | Result |
|-------|-------|-------------|--------|
| 1: Baseline | Scipy on three branches | exp005, 010, 026 | ✅ All branches reproduced |
| 2: Chaos exploration | Fine sweep [0.52, 0.60] with scipy | exp100-134 | ⚠️ Found apparent fractality |
| 3: Solver validation | Fourier 1-mode on same region | exp150-154 | 🔍 **BREAKTHROUGH**: chaos vanished |
| 4: Complete mapping | Fourier sweep u_offset [-1.5, 1.5] | exp155-220+ | ✅ True basin structure revealed |
| 5: Perturbations | Amplitude/phase effects | exp264-269 | 🟡 Ongoing (agent3) |

## Why This Is Important

1. **Demonstrates multi-solver validation necessity:** Single-solver studies can reach wrong conclusions
2. **Residual floor resolution principle:** Numerical resolution determined by solver, not physics
3. **Non-obvious bifurcation structure:** Three branches with overlapping non-monotone parameter regions
4. **Publishable finding:** "Fourier Spectral Resolution of Bifurcation Basin Structure in Nirenberg PDE"

## Next Recommended Actions

✅ **High priority:** Bifurcation parameter continuation (K_amplitude) to trace basin evolution  
✅ **Medium:** Fine-grain exact transition points (u_offset ≈ -0.50, 0.45, 0.62)  
✅ **Medium:** Test 2-3 Fourier modes (can we beat 5.55e-17?)  
✅ **Low:** 2D basin mapping (u_offset × amplitude space)

## Learned Constraints

❌ **Scipy residual floor:** Hard limit at 3.25e-12 for tol=1e-11  
❌ **Tolerance tuning (scipy):** No improvement beyond tol=1e-11 (looser = worse, tighter = crash)  
❌ **Fine u_offset sweeps (scipy):** Region [0.52, 0.60] is irreducible with scipy—use Fourier instead
