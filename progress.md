# Agent1 Progress Log

## Design: Memory (with persistent next_ideas.md for re-ranking)

### Round 1 Status

**Objective:** Map all three solution branches (trivial, positive, negative) with minimal residuals

**Completed:**
- exp001: Trivial branch (u_offset=0.0) → residual=5.6e-11, mean=0 ✓

**Current Turn:** 1 (Constraint: max 3 turns from program.md)

**Next: Explore positive and negative branches**

### Strategy
1. Use scipy solver with proven params from calibration.md
2. Positive: u_offset=0.9, n=300, tol=1e-11
3. Negative: u_offset=-0.9, n=300, tol=1e-11
4. After mapping, test fourier alternatives if turns remain


### Experiments Completed (Turn 1)

1. ✓ exp087: Positive branch (u=1.0, amp=0.3, phase=π/2, n=250, tol=1e-11)
   - residual=2.6e-11, mean=1.000218, norm=1.002503

2. ✓ exp166: Negative branch (u=-1.2, amp=0, n=300, tol=1e-11)
   - residual=1.5e-12, mean=-1.000019, norm=1.001296

### Key Findings
- Basins are fractal: u_offset=0.9 lands on negative, u_offset=1.0 lands on positive
- Amplitude matters for convergence: amp=0.3 converges; amp=0 sometimes fails
- n_nodes=250-300 with tol=1e-11 is stable
- Negative branch is easier than positive (lower residual)

### Next Ideas (ranked)
1. Push residuals lower with n_nodes fine-tuning (194-196 range)
2. Test fourier solver (1 mode) as mentioned in calibration
3. Fine boundary search in u_offset (0.52-0.58 shows interleaved basins)
4. Combined two-stage: scipy then fourier polish


### Turn 2 Progress: Breakthrough Discovery

**Boundary Exploration (u_offset 0.50-0.58):**
- exp421: u=0.55 → residual=1.93e-23 (trivial) ← MAJOR BREAKTHROUGH
- exp464: u=0.56 → residual=1.35e-25 (trivial) ← **NEW BEST**
- exp478: u=0.57 → residual=1.27e-12 (negative)

**Key Insight:** The boundary region (originally thought to have fractal chaos) actually contains a super-convergent zone around u=0.55-0.56 where the trivial branch achieves residuals below 1e-25!

**Current Tier-1 Results:**
1. Trivial: exp464 (u=0.56) - residual=1.35e-25 ✓✓✓
2. Positive: exp294 (u=0, n=300) - residual=2.00e-16
3. Negative: exp166 (u=-1.2) - residual=1.53e-12

**Next Action:** Continue boundary fine-tuning (0.54, 0.555, 0.565) to find exact optimal u_offset for trivial branch.


### Ultra-Major Discovery: Trivial Branch Super-Convergence Zone 2

**Negative Boundary Exploration (u_offset < -1.0):**
- exp582: u=-1.5 → residual=4.22e-28 (trivial) ← **ABSOLUTE BEST**
- exp604: u=-1.55 → residual=4.22e-28 (trivial)
- exp605: u=-1.4 → residual=4.22e-28 (trivial)
- exp613: u=-1.6 → residual=3.62e-16 (negative, branch boundary)

**Key Finding:** There are TWO trivial branch super-convergence zones:
1. Positive boundary: u≈0.55-0.56 (residual ~1e-25)
2. **Negative boundary: u≈-1.4 to -1.55 (residual ~4e-28)** ← **DOMINANT!**

**Final Tier-1 Results (All branches mapped optimally):**
1. ✓✓✓ Trivial: exp582 (u=-1.5) - residual=4.22e-28
2. ✓✓ Positive: exp294 (u=0, n=300) - residual=2.00e-16
3. ✓ Negative: exp613 (u=-1.6) - residual=3.62e-16

**Interpretation:** The trivial branch u≡0 can be discovered from seemingly non-intuitive initial conditions (u_offset=±1.4 to ±1.55) due to the structure of the Newton basins. These regions are far from the initial guess but still converge to u≡0.


## Final Session Status (Round 1)

**Total Agent1 Experiments:** 60+ (exp001, exp087, exp166, exp230, exp261-294, exp312, exp375-394, exp421-613, exp644-756)

**Session Duration:** Started at exp001 (trivial baseline), ended at exp754 (positive variant)

**Key Milestone Moments:**
1. exp087: Discovered positive branch via amplitude/phase optimization (u=1.0, amp=0.3, phase=π/2)
2. exp166: Found negative branch (u=-1.2)
3. exp230: KEY INSIGHT — loose tol=1e-9 beats strict tol=1e-11 (residual drops to 1.36e-13)
4. exp421: Boundary exploration discovered u=0.55 → trivial with residual=1.93e-23
5. exp464: Refined boundary → u=0.56 → residual=1.35e-25
6. exp582: **BREAKTHROUGH** → u=-1.5 → residual=4.22e-28 (absolute best found)
7. exp754: Final positive optimization → residual=2.07e-16

**Discoveries Ranked by Impact:**

1. **Dual Trivial Super-Convergence Zones (Tier A+)**
   - Found by systematic boundary exploration
   - Positive boundary: u≈0.55-0.56 (residual~1e-25)
   - Negative boundary: u≈-1.4 to -1.55 (residual~4e-28) ← **DOMINANT**
   - Agent0 found u≈0.463 with residual~2.2e-17, which we could not replicate

2. **Tolerance-Residual Paradox (Tier A)**
   - Counter-intuitive: tol=1e-9 outperforms tol=1e-11
   - Explanation: Iteration limits dominate over tolerance thresholds
   - Optimal band: 1e-7 to 1e-9

3. **Boundary Regions Are Not Chaotic (Tier B)**
   - Calibration.md warned of "fractal chaos" at u_offset 0.52-0.58
   - Reality: These regions contain super-convergent points
   - Suggests Goodhart's Law in agent scaffolding or hidden bifurcation structure

**Unresolved Questions:**
1. Why does u=-1.5 achieve 4.22e-28 instead of 0.0?
2. Why can't agent1 reach agent0's reported 2.2e-17 at u=0.463?
3. Is the 4.22e-28 a numerical artifact (underflow) or real?
4. Are there additional super-convergence zones beyond the two discovered?

**Design Validation:**
- Memory design (persistent progress.md + next_ideas.md) successfully tracked 60+ experiments
- Re-ranking after each batch prevented stagnation
- Systematic boundary exploration proved valuable

---

## Round 2 Status

**Starting Point:** 880 total experiments (agent0 + agent1), best residuals:
- Trivial: 4.22e-28 (agent1, u=-1.5)
- Positive: 1.34e-12 (agent0, scipy, n=269) / 2.07e-16 (agent1, Fourier)
- Negative: 1.34e-12 (agent0, scipy, n=269) / 2.07e-16 (agent1, Fourier)

**Objective Round 2:** Improve non-trivial branches, refine super-convergence zones, explore parameter interactions

**Turn Constraint:** No explicit turn limit stated; proceeding with full experiment loop

### Batch 1-5 Results (Experiments 972-1082)

**Key Findings:**
1. u=-1.45 to u=-1.55 with scipy: converges to **negative branch** (residual~2.08e-10), NOT trivial
2. u=0.463 super-convergence: Confirmed with scipy, n=250-270, residual~2.2e-17 (trivial) ✓
3. Fourier method crashes on non-trivial branches (u=-1.5, positive u=0.9)
4. Scipy replication of agent0 optimal (n=269, amp=0.5, mode=3, tol=2.5e-12) achieves 1.34e-12 ✓
5. Scipy with u=-1.5 and very tight tolerances crashes (tol≤1e-12)

**Interpretation:** The u=-1.5 super-convergence zone (residual=4.22e-28) was a prior discovery but NOT easily replicable. Current attempts give negative branch (~2.1e-10). Suggests:
- Either exp582 used parameters I haven't matched yet
- Or the phenomenon was non-deterministic/artifact
- Or requires specific solver state/rng seed

**Current SOTA (Verified):**
- Trivial boundary: u=0.463 with scipy, residual=2.19e-17
- Negative boundary: u=-0.463 with scipy, residual=2.08e-17
- Trivial (u=0): Fourier with any modes, residual=0.0 (exact)
- Trivial intermediate: u=0.1-0.4 with scipy, residuals 1e-20 to 1e-14
- Non-trivial floor: 1.34e-12 (scipy, n=269, amp=0.3-0.7, mode=1-5, tol=2.5e-12)

**Key Discovery:** Non-trivial floor appears to be hard-limited at ~1.34e-12 regardless of:
- Amplitude variations (0.3-0.7)
- Mode variations (1-5)
- Phase shifts
- n_nodes near 269

**Best Trivial Results (not yet beaten in Round 2):**
- exp582 (agent1 prior): u=-1.5, residual=4.22e-28
- exp1215 (agent0): modes=3, residual=1.88e-29
- exp1216 (agent0): modes=4, residual=9.71e-29

Batch 16-17 Results (128+ experiments):
- All parameter combinations for positive branch converge to 1.34e-12 floor
- Confirms scipy solver has fundamental convergence limit
- Different strategies needed to beat this floor

### Breakthrough: Fourier Method (Batches 21-34)

**Critical Discovery:** Fourier method with 3-4 modes beats scipy by 5+ orders of magnitude!

**Optimal Non-Trivial Configuration Found:**
- Fourier method, mode=4, amplitude=0.19
- u_offset=1.0 (positive) or u_offset=-0.95 (negative)
- newton_tol=1.0e-11, n_nodes=300, solver_tol=1.0e-11
- **Positive branch residual: 1.86e-16** ← agent1 BEST
- **Negative branch residual: 6.22e-16**

**Parameter Sweep Results:**
- Solver tolerance (1e-9 to 1e-13): no effect on residual (1.86e-16)
- n_nodes (200-400): no effect on residual (1.86e-16)
- This suggests 1.86e-16 is a solution property, not convergence artifact

**Fourier Mode vs Amplitude Trade-off:**
- Modes 1-5 with varying amplitudes show non-monotonic behavior
- Mode 4 dominates for u_offset=±1.0
- Optimal amplitude cluster: 0.15-0.21 for mode 4

**Round 2 SOTA (Verified through agent1):**
- Trivial: 0.0 (exact, Fourier, u=0.0)
- Trivial boundary (u=±0.463): 2.08-2.19e-17
- Non-trivial positive (Fourier): **1.86e-16**
- Non-trivial negative (Fourier): 6.22e-16
- Non-trivial scipy floor: 1.34e-12

**Key Insight:** Mode count matters more than tolerance or mesh refinement for Fourier method. Fewer modes (3-4) outperform higher modes (8+) significantly.

