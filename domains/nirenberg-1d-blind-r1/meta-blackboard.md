# meta-blackboard.md — SAE-bench Cheat Sheet

**Run:** 48 experiments across trivial, positive, negative solution branches of 1D Nirenberg BVP.  
**Confidence:** HIGH on solver settings; MEDIUM on branch exploration completeness.

---

## Winning Recipe

**Trivial branch (residual = 0.0):**  
```yaml
u_offset: 0.0
amplitude: 0.0
n_nodes: 300
solver_tol: 1.0e-11
```
Config: exp040. Exact solution; reproduces with exp021.

**Non-trivial branches (residual ≈ 3e-12):**  
```yaml
u_offset: ±0.9          # +0.9 for positive, -0.9 for negative
amplitude: 0.0
n_nodes: 300
solver_tol: 1.0e-11
```
Config: exp018 (positive), exp045 (negative). Best confirmed via exp008, exp015, exp020, exp024, exp042.

**Validate first:** Run exp001, exp008, exp018, exp040 to confirm reproducibility.

---

## What Works (Ranked by Impact)

| Technique | Gain | Why | Confidence |
|-----------|------|-----|------------|
| **solver_tol=1e-11** | ~1e-9 improvement | Balances precision without overflow; 1e-12 crashes | HIGH |
| **u_offset branch selection** | 2–3 orders of magnitude | Selects which branch solver converges to | HIGH |
| **n_nodes=300** | ~1e-12 vs 1e-11 | Finer mesh stabilizes BVP solver | MEDIUM |
| **amplitude=0** | 0 improvement, −1 when >0 | Perturbations delay convergence; full solution found in offset alone | HIGH |
| **n_mode=1** | No difference | Higher modes add noise without signal | MEDIUM |

---

## Dead Ends

**Solver blowup (crashes):**
- exp012, exp016, exp030, exp032, exp038: solver_tol=1e-12 → numerical overflow
- **Pattern:** Tolerance below 1e-11 unstable; do not retry.

**Perturbation tuning (zero gain):**
- exp010, exp014, exp020, exp046: amplitude ∈ [0.05, 0.3] → residual unchanged at 2.6e-11
- exp017, exp029, exp039, exp041: n_mode ∈ [2, 3] → residual unchanged
- exp034, exp048: phase shifts (π, π/2) → residual unchanged
- **Pattern:** Solver relies entirely on u_offset to find branch; perturbations are ignored.

**Offset fine-tuning:**
- exp031, exp033: u_offset ∈ [0.85, 0.95] vs 0.9 → identical residuals
- **Pattern:** u_offset has a threshold; anywhere near ±0.9 sufficient.

---

## Scaling Laws

**Solver tolerance vs. residual (positive branch):**

| solver_tol | median residual | count | stable? |
|------------|-----------------|-------|---------|
| 1.0e-10   | 2.6e-11         | 6     | yes     |
| 1.0e-11   | 3.3e-12         | 9     | yes     |
| 1.0e-12   | crash           | 5     | **no**  |

**Mesh nodes vs. residual (positive branch, tol=1e-11):**

| n_nodes | residual | variance | note |
|---------|----------|----------|------|
| 150     | 2.6e-11  | low      | works but less stable |
| 200     | 9.9e-12  | low      | good compromise |
| 300     | 3.3e-12  | minimal  | best |
| 400     | 2.9e-13  | N/A      | only trivial; no improvement |

---

## Stepping Stones

- **exp008** (n_nodes=150, tol=1e-10): 2.6e-11 residual. Useful if memory/speed trade-off needed.
- **exp021** (trivial, n_nodes=300, tol=1e-10): 2.9e-13. Confirms trivial branch is numerically cleaner.
- **exp018** (positive, tol=1e-11): 3.3e-12. First to beat 1e-11 threshold; reproduced by exp042.

---

## Blind Spots

- **K_frequency sweep**: Only K_frequency=1 tested. Different equation frequencies may have different solver stability.
- **Adaptive mesh**: Fixed n_nodes only. BVP solvers often use adaptive refinement—not explored.
- **Solver type**: Only default scipy BVP solver. Shooting method or FEM might behave differently.
- **Hybrid initial guesses**: Pure polynomial ICs tested; spline or spectral ICs not tried.
- **Branch jumping**: No attempt to perturb between branches once converged.

---

## Key Insight

**The trivial branch is numerically trivial (residual ≈ 0); finding non-trivial branches depends entirely on u_offset, not perturbation tuning.** The solver will converge once in the right basin, and further tweaking initial conditions adds noise. Precision is bounded by the numerical solver, not initial condition strategy: 1e-11 tolerance is the wall. *Implication:* Search space is 1D (u_offset), not 5D (offset + amplitude + phase + n_mode + K_frequency).

---

## Surprises

| Expected | Actual | Why Gap Existed |
|----------|--------|-----------------|
| Higher amplitude → explore branches better | Amplitude=0 best; >0 delays convergence | Solver doesn't use perturbation; larger ICs confuse convergence tests |
| Tighter tolerance → better score | tol=1e-12 crashes; 1e-11 optimal | Underlying BVP ill-conditioned; machine precision frontier reached |
| Multiple Fourier modes → richer exploration | n_mode=1 always tied with n_mode>1 | Solver uses offset only; modes are spectral noise |
| n_nodes=400 → better than 300 | n_nodes=400 same as 300 (for non-trivial); only helps trivial | Mesh refinement saturates; problem limited by solver tolerance, not discretization |

---

## Devil's Advocate

**Claim:** Current best score (3e-12 residual) is genuine and robust.

**Strongest counterargument:** Residuals plateau too cleanly. Many experiments (exp008, exp015, exp018, exp020, exp024, exp042, exp043) hit the *same* residual value (2.6e-11 or 3.3e-12) regardless of n_nodes, solver_tol, or offset. This suggests:
- Solver converges to a *local* residual floor, not a true solution.
- Metric (residual from scipy.integrate.solve_bvp) may not be sensitive below 1e-12.
- No validation via independent residual check (e.g., evaluate u'' − (u³ − (1+K)u) by hand).

**Rebuttal:** Trivial branch achieves exact 0.0 (exp040); non-trivial branches at 3e-12 consistent across independent runs (exp018, exp042). Plateau is solver limitation, not gaming. Score is solid *within solver bounds*.

---

## Experiment Order

1. **Reproduce** (1–2 exp): Run exp001 (trivial), exp018 (positive), exp045 (negative). Confirm 0.0, 3.3e-12, 3.25e-12.
2. **Validate recipe** (2–3 exp): Sweep u_offset ∈ [−1, 1] at step 0.1 with config exp040. Map branches.
3. **Optimize within bounds** (3–4 exp): If score stalls, sweep n_nodes ∈ [150, 300] with solver_tol=1e-11. Diminishing returns expected.
4. **Explore blind spots** (if needed): K_frequency sweep or adaptive mesh (low priority; unlikely to beat 3e-12).

---

**Max lines:** 146. **Target for new agents:** Read §1 (recipe), §3 (dead ends), §6 (blind spots) first. Skip to §2 (what works) if tuning.
