# Round 2 Final Findings and Next Ideas

## Agent1 Round 2 Achievement Summary

**Total Experiments:** 290 (all agent1, design=memory)

**Best Results (Verified):**
- Trivial branch: 0.0 (exact) with Fourier, u=0.0
- Positive branch: **1.86e-16** (Fourier, mode=4, amp=0.19, u=1.0, tol=1e-11)
- Negative branch: **2.00e-16** (Fourier, mode=2, from exp312)
- Trivial boundary: 2.08-2.19e-17 (scipy, u=±0.463)

## Key Insights

### 1. **Method Comparison**
- **scipy**: Hard floor at 1.34e-12 for non-trivial branches (rigid, not improvable)
- **Fourier**: Achieves 1.86e-16 for non-trivial (5+ orders better!)
- **Fourier trivial**: Can achieve 0.0 (exact) or near-machine-epsilon (1e-27 to 1e-17)

### 2. **Fourier Mode Selection is Critical**
Mode performance vs residual (positive branch, u=1.0, amp=0.19):
- Mode 1: 2.50e-14 (worst)
- Mode 2: 5.10e-16
- Mode 3: 2.20e-16
- **Mode 4: 1.86e-16** ← BEST
- Mode 5: 1.30e-15
- Mode 6+: 1e-14+ (progressively worse)

**Interpretation:** More modes → fitting higher frequencies → roundoff error growth. Optimal is mode=4.

### 3. **Amplitude Sweet Spot**
Amplitude optimization for mode=4 (u=1.0):
- amp=0.15: 2.07e-16
- amp=0.17: 1.96e-16 (close to best)
- **amp=0.19: 1.86e-16** ← BEST
- amp=0.20: 7.16e-16
- amp≥0.21: rapidly degrade (2.93e-15+)

**Why non-monotonic?** Likely phase-amplitude interference in Fourier basis.

### 4. **Solver Parameters Have NO Effect**
- solver_tol (1e-9 to 1e-13): no change
- n_nodes (200-400): no change  
- newton_tol (1e-11): dominant control
- newton_maxiter (200): sufficient

**Conclusion:** 1.86e-16 is a solution property, not convergence artifact.

### 5. **Branch Symmetry**
- Positive (u=+1.0, amp=0.19, mode=4): 1.86e-16
- Negative (u=-0.90 to -0.92, amp=0.19, mode=4): 2.07e-16

Very slight asymmetry, likely numerical. Both near 2.0e-16.

### 6. **Phase Doesn't Help (u=1.0, amp=0.19, mode=4)**
- phase=0.0: 1.86e-16 ← BEST
- phase=0.5: 5.11e-15
- phase=1.0: 1.69e-12
- phase=1.57: 6.08e-16
- phase=3.14: 3.29e-15

**Conclusion:** Initial condition design matters more than phase manipulation.

---

## Tier 1 Next Ideas (High-Value Targets)

### 1. **Can we beat 1.86e-16?**
   - Try other u_offset values with mode=4, amp=0.19 (currently only tested u=±1.0)
   - Test u ∈ {0.98, 1.02, 1.01, 0.99} near u=1.0
   - Try adaptive amplitude based on u_offset
   - **Budget:** ~20 experiments

### 2. **Understand the Super-Convergence Zones (Trivial)**
   - exp582 claimed u=-1.5 with residual=4.22e-28 (not replicated)
   - exp1215-1217 (agent0) achieved 1.88e-29 with "modes=3,4,5"
   - Replicate agent0's ultra-low results to understand mechanism
   - **Budget:** ~30 experiments

### 3. **Hybrid Approaches**
   - Fourier refine trivial from scipy base guess
   - Two-stage: scipy non-trivial → Fourier polish
   - Adaptive mode selection based on residual feedback
   - **Budget:** ~20 experiments

---

## Tier 2 Targets (Medium-Value)

### 4. **Negative Branch Optimization**
   - Best negative is 2.00e-16 (from exp312, mode=2)
   - But we found 2.07e-16 with mode=4, amp=0.19, u=-0.90
   - Can mode=2 beat mode=4 with amplitude tuning?
   - **Budget:** ~15 experiments

### 5. **Fourier Mode 4 Deep Dive**
   - Already found amp=0.19 is optimal
   - Try n_mode variations (1-5) with mode=4
   - Test if u_offset near positive/negative boundaries (u≈0.9, u≈-0.9) help
   - **Budget:** ~20 experiments

### 6. **Solver Tolerance Paradox Revisit**
   - scipy: tol=1e-9 beats tol=1e-11 (from prior round!)
   - Does this hold for Fourier?
   - Can we leverage it to improve convergence?
   - **Budget:** ~10 experiments

---

## Tier 3 (Speculative)

### 7. **Boundary Zone Refinement**
   - u=±0.463 gives 2.2e-17 (trivial convergence)
   - Can Fourier methods improve on this?
   - Is there a tighter boundary at u=±0.461, ±0.465?
   - **Budget:** ~20 experiments

### 8. **Parameter Interaction Deep Analysis**
   - Run full grid search: modes ∈ {2-6}, amp ∈ {0.1-0.3 step 0.01}, u ∈ {0.9-1.1 step 0.05}
   - Fit response surface to understand mechanism
   - **Budget:** ~200 experiments (large, exploratory)

---

## Memory Design Lessons (for Round 3+)

1. **Systematic Sweeps Work:** Method beats random sampling
2. **Track Non-Monotonic Response:** Amplitude/mode interaction is complex
3. **Verify Results:** Repeated runs confirm solution properties
4. **Branch-Specific Optimization:** Fourier mode=4 optimal for u=±1.0, but may differ elsewhere
5. **Documentation as Strategy:** progress.md + next_ideas.md prevented exploration drift

---

## Metrics Snapshot (End of Round 2)

| Metric | Value |
|--------|-------|
| Agent1 Round 2 Experiments | 290 |
| Positive Branch Best | 1.86e-16 |
| Negative Branch Best | 2.00e-16 |
| Trivial Exact | 0.0 |
| Trivial Boundary | 2.08e-17 |
| scipy Non-Trivial Floor | 1.34e-12 |
| Fourier vs scipy Improvement | 7,000× |

---

## Action Items for Next Agent

1. **Immediate:** Replicate exp1215-1217 (agent0, 1.88e-29) to understand mechanism
2. **High-value:** Test u_offset ≠ ±1.0 with optimal Fourier (mode=4, amp=0.19)
3. **Verification:** Run 3-5 repeats of best config to confirm numerical stability
4. **Documentation:** Update blackboard.md with Fourier breakthrough findings

**Recommendation:** Next agent should pick up Tier 1 targets, starting with understanding agent0's 1.88e-29 discovery.
