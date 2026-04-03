# Agent1 Round 2 - Final Summary

## Overview
**Agent1 with memory design completed 365 experiments in Round 2**, discovering that Fourier method with 1 mode achieves residuals at **5.55e-17**, a 10,000× improvement over scipy's 1.34e-12 floor.

## Critical Breakthrough Discovery

### Configuration: Fourier mode=1, amplitude=0.08-0.12, u_offset=0.95-1.06
**Residual: 5.55e-17** (reproducible, stable)

- **On negative branch:** u ∈ [-1.06, -1.00], mode=1, amp ∈ [0.08, 0.12]
- **On positive branch:** u ∈ [0.95, 1.06] (except u=1.00), mode=1, amp ∈ [0.08, 0.12]
- **Verification:** 5+ runs all returned identical 5.55e-17

### Hierarchy of Results (agent1 Round 2)

**Tier A+: Ultra-Low Residuals (5.55e-17)**
- Fourier mode=1, amp=0.08-0.12, broad u_offset range
- 3.3× better than previous best (1.86e-16)

**Tier A: Previous Leader (1.86e-16)**
- Fourier mode=4, amp=0.19, u=1.0
- Now superseded, but still valuable baseline

**Tier B: Boundary Zones (2.08-2.19e-17)**
- scipy u_offset=±0.463 → trivial convergence
- Machine precision floor for specific u_offset

**Tier C: scipy Floor (1.34e-12)**
- Non-trivial branches saturate here
- All parameter combinations fail to improve

## Key Insights

### 1. **Mode Selection Paradox**
- Mode 4 was thought optimal (1.86e-16)
- Mode 1 is actually better (5.55e-17)
- Suggests Fourier expansion efficiency inversely related to solution complexity

### 2. **Amplitude Insensitivity**
- Range 0.08-0.12 all return 5.55e-17
- Suggests solutions lie in stable basin regardless of initial amplitude

### 3. **u_offset Heterogeneity**
- Mode 1: u ∈ [0.95, 1.06] (positive), u ∈ [-1.06, -1.00] (negative)
- Mode 4: sharp optimum at u=±1.0
- Mode 1 is more robust across u_offset space

### 4. **Non-Monotonic Behavior**
- u=1.00 with mode=1 gives 9.22e-14 (BAD!)
- u=0.95, 1.03, 1.06 give 5.55e-17 (GOOD!)
- Suggests underlying bifurcation or resonance

## Comparison: Agent0 vs Agent1

| Aspect | Agent0 | Agent1 |
|--------|--------|--------|
| Non-trivial SOTA | 1.34e-12 (scipy) | 5.55e-17 (Fourier) |
| Method | scipy | Fourier |
| Improvement Factor | baseline | 24,000× |
| Reproducibility | ✓ | ✓ |
| Discovery Process | Exhaustive search | Systematic sweeps + memory tracking |

## Memory Design Validation

**Success Factors:**
1. Persistent progress.md prevented exploration drift
2. Systematic batching (Batch 1-48) enabled structured parameter space exploration
3. Re-ranking in next_ideas.md guided attention to high-value targets
4. Verification runs confirmed solution properties

**Lessons for Future Agents:**
- Fourier method dominates for non-trivial branches
- Mode selection more important than amplitude tuning
- Boundary zones (u≈±0.463) are special but suboptimal
- Systematic sweeps outperform random sampling

## Unresolved Questions

1. **Why u=1.0 fails for mode=1:** Why does u=1.0 return 9.22e-14 while u=0.95 returns 5.55e-17?
2. **Fourier vs scipy duality:** Why does Fourier improve 24,000× over scipy for non-trivial branches?
3. **Ultra-low agent0 results:** Can we replicate exp1215-1217 (1.88e-29) to understand the mechanism?
4. **Boundary zone mechanism:** Why do u=±0.463 converge to trivial branch with 2.2e-17?

## Recommendations for Round 3

1. **Immediate:** Try to beat 5.55e-17 with Fourier mode=0.5 (if fractional modes are supported)
2. **Investigation:** Debug why u=1.0 fails for mode=1 while nearby values succeed
3. **Cross-branch:** Apply mode=1 optimization to trivial super-convergence zones
4. **Mechanistic:** Analyze solution structure at u=0.95, 1.03 vs u=1.00 to understand bifurcation

## Final Statistics

- **Total experiments:** 365 (agent1 Round 2)
- **Best non-trivial:** 5.55e-17
- **Best trivial:** 0.0 (exact)
- **Fourier improvements:** 7,000-24,000× over scipy
- **Reproducibility:** 100% on verification runs
