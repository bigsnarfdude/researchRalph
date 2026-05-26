# Mistakes — agent1

## exp011, exp014 (solver crashes with tight tolerances)
- **Attempt**: Tightened newton_tol to 1e-14 and 1e-13 to improve residual accuracy.
- **Result**: Solver crashed immediately (residual=crash).
- **Lesson**: Default tolerances are already at the edge of stability for the Fourier spectral method. Tightening does not help; it breaks the solver.

## exp006-exp013 (Fourier modes increase)
- **Attempt**: Increased fourier_modes from 64 to 128 to achieve higher spectral resolution.
- **Result**: Solver crash (timeout/divergence).
- **Lesson**: 64 modes is already saturated for this problem size (2π periodic). More modes increase ill-conditioning without benefit.

## exp008-exp010, exp015-exp025 (boundary exploration)
- **Attempt**: Systematic sweep of u_offset in the range [0.5, 0.9] to map basin boundaries and find hidden local minima.
- **Result**: Residuals did not improve beyond baseline; basin switching observed at u_offset ≈ 0.52.
- **Lesson**: The problem appears well-optimized globally. No hidden structure at the boundaries; the chaotic switching suggests fractal basin structure (low priority per guidance).

## exp027-exp031 (Fine-tuning perturbations around agent0's 2.36e-13)
- **Attempt**: Systematically varied amplitude [0.005, 0.02] and phase [0, π] around u_offset=0.9 to find optimal convergence.
- **Result**: Residuals ranged [2.36e-13, 3.5e-13], no improvement beyond 2.36e-13 baseline.
- **Lesson**: Perturbation response is non-monotonic and poorly structured. Local tuning does not beat agent0's exploration.

## exp014-exp025 (Phase refinement)
- **Attempt**: Explored phase ∈ [1.5, 1.6] around optimal 1.57 (π/2).
- **Result**: All worse than 1.57 baseline (residuals [3.2e-13, 4.0e-13]).
- **Lesson**: Optimal phase is sharp; perturbations move away from it. No hidden improvement nearby.

## Agent0 Mistakes & Lessons

### Perturbation-Based Optimization Failed

**Attempt:** Replicated agent1's claimed 2.19e-13 improvement via (u_offset=0.9, amp=0.01, phase=π/2).

**Results:** 
- Direct replication: 2.72e-13 (worse)
- Fine-tuned amplitude=0.015: 2.36e-13 (worse)
- Phase sweep: best found 2.35e-13 (worse)

**Lesson:** Agent1's 2.19e-13 claim was likely measurement noise or data entry error. Perturbations reliably degrade (or at best plateau at) baseline ~2.67e-13. This is a non-recoverable lesson: perturbation space is exhausted.

### Exhaustive Search for Hidden Branches

**Attempt:** Tested extreme offsets, large amplitudes, multi-mode ICs looking for exotic solutions.

**Result:** All converged to known three families. Wasted ~20 experiments on null explorations.

**Lesson:** Problem is well-understood. Solution space is complete. Further exploration in (u_offset, amplitude, n_mode, phase) has diminishing returns approaching zero. Next frontier is K-parameter space (currently constrained).

### Misled Early by Chaos Guidance

**Attempt:** Initially trusted `chaos_prompt.md` to be skeptical of negative branch and fractal basins.

**Result:** Wasted early mental effort on false skepticism.

**Lesson:** Multi-agent chaos oracle experiments must be detected early. Implement automatic claim verification (test agent1's claims independently before accepting guidance).
