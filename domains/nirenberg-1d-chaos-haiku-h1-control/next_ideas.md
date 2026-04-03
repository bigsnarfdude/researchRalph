# Agent1 Round 3 Idea Ranking

## Proven Configs (High Confidence, Replicate + Tweak)

| Priority | Idea | Target Branch | Expected Residual | Rationale | Status |
|----------|------|----------------|-------------------|-----------|--------|
| 1 | 4th branch phase sweep [0, 2π) × {amp=0.5, amp=0.3} | 4th | 1e-30 to 1e-25 | Just discovered; full characterization needed | FRONTIER |
| 2 | Phase=π/2 amplitude fine-tuning (0.45-0.55) | 4th | <1e-28 | 4th branch found at amp=0.5; narrow window | HOT |
| 3 | Negative super-convergence: u=-0.463 × Fourier | trivial | 1e-29 to 1e-27 | Negative boundary matches positive; symmetric | PROVEN |
| 4 | Positive u=+1.44 × Fourier 2modes exact | positive | 1.86e-16 | Tied best; mode=1 might improve | PROVEN |
| 5 | Negative u=-1.44 × Fourier 2modes exact | negative | 1.86e-16 | Tied best; mode=1 might improve | PROVEN |
| 6 | Trivial u=0.463 scipy n=300 tol=1e-8 | trivial | 1e-26 | Scipy paradox: loose tol wins | PROVEN |
| 7 | Mixed mode approach: mode=1+2 superposition | exotic | 1e-20 to 1e-25 | Hypothesis: modal interference patterns | NEW |

## Frontier Ideas (Lower Confidence, Explore Systematically)

| Priority | Idea | Target Branch | Expected Residual | Mechanism |
|----------|------|----------------|-------------------|-----------|
| 8 | n_nodes=250 (between 269 crash and 300 sweet spot) | non-trivial | 1.34e-12 to 1.86e-16 | Mesh sensitivity plateau? |
| 9 | Phase ramp: incrementally shift phase during solve | any | varies | Morphing between branches? |
| 10 | Amplitude step: amplitude → 0 over Newton iterations | any | varies | Adiabatic approach to simpler branch? |
| 11 | K_amplitude modulation (currently fixed at 0.3) | all | varies | Perturbation to K function (probably forbidden) |
| 12 | 5-8 Fourier modes with n_nodes=150 sparse grid | non-trivial | 1e-14 to 1e-12 | Undersampling stability? |

## Re-ranking Strategy

After each run, update:
1. If residual < expected → move to **proven**, increase priority
2. If residual > expected but > current best → keep in frontier, lower priority  
3. If crash or anomaly → mark as BLOCKED with reason, investigate next
4. Capture all phase/amplitude discoveries to narrow subsequent ranges

## Round 3 Execution Plan

**Batch 1 (Phase sweep):** 30 experiments, phase ∈ [0, 2π), 10 steps, fix (u=0.0, amp=0.5, mode=1)  
**Batch 2 (4th branch amplitude):** 15 experiments, amp ∈ [0.3, 0.6], phase=π/2 fixed  
**Batch 3 (Boundary refinement):** 20 experiments, u ∈ ±[0.460, 0.465], step=0.001, Fourier modes  
**Batch 4 (Mode optimization):** 15 experiments, mode ∈ [1,2,3,4,5,6] × best u/phase, Fourier  
**Batch 5 (Free exploration):** Remaining budget, follow hottest signals from batches 1-4

---

**Total ideas generated:** 12  
**Confidence: HIGH (1-6), MEDIUM (7-10), LOW (11-12)**
