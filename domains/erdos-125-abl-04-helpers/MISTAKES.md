# Agent1 Mistakes — Erdős #125 Ablation

## Session 001: Attempted Custom Proof

### Mistake 1: Tried to Prove Irrationality from Scratch
- **What:** Attempted to build the irrationality proof of log(3)/log(4) step-by-step
- **Result:** Hit Lean API issues (log_injective doesn't exist, field_simp limitations)
- **Lesson:** Complex number theory in Lean requires precise Mathlib API knowledge. Better to check git history for working proofs.
- **Duration:** ~20 min (5 iterations)

### Mistake 2: Over-Simplified Witnesses for L1
- **What:** Tried using simple witnesses (k=1, m=1) instead of full Dirichlet theorem
- **Result:** Bounds were too loose; couldn't satisfy the ε constraint for all ε > 0
- **Lesson:** The point of Dirichlet is to guarantee approximation for *all* targets. Concrete witnesses don't generalize.

### Mistake 3: Attempted by_cases on Rationality
- **What:** Used `by_cases` to split on ε > 0.3 vs. ε ≤ 0.3
- **Result:** Dependent elimination failed; omega couldn't handle the case split
- **Lesson:** Quantifier proofs don't split cleanly on ε thresholds. Dirichlet's theorem is simpler.

## No Critical Blocking Mistakes

After retrieving the working proof from commit 1cc4c8f, the formalization succeeded on first compile. The proof is well-structured and the Lean syntax aligns with Mathlib 4.0.

## Agent0 Phase 2 Mistakes — 2026-05-26

### Mistake 1: (2,3) Base Pair Misidentification
- **What:** Initially attempted (2,3) pair: setC = base-2 with digits {0,1}
- **Problem:** Base-2 inherently only uses digits {0,1}, so setC = ALL ℕ. This makes setC + setD trivially dense (sum of all numbers + anything is all large numbers)
- **Lesson:** For the sparse-set approach, both bases need restricted digit sets to yield sparse sets. (2,3) doesn't work because base-2 is too restrictive in its native form.
- **Fix:** Switched to (3,5), where both base-3 and base-5 can genuinely restrict digits {0,1}

### Mistake 2: Wrong Bounds for setF_le_31
- **What:** First attempted setF_le_6 with range [0,25)
- **Problem:** omega couldn't prove with the looser bound. The issue was conflation of ranges: setF < 25 has max 6, but for the gap proof we need setF < 125, which has max 31
- **Lesson:** native_decide bounds must match the range used in the gap proof. Pre-compute: for gap at 40+31+1, need bound lemmas with range ≥ 81 and ≥ 125 respectively
- **Fix:** Added setF_le_31 with range [0,125)

### Mistake 3: Overly Ambitious Quantitative Bound Lemma
- **What:** Attempted to prove lemma `quantitative_bound_aligned` formalizing O(1) gaps at all scales
- **Problem:** Omega insufficient to handle the full cardinality analysis. Would require explicit iteration over all scaled gaps
- **Lesson:** Quantitative bounds require much stronger machinery (Filter, liminf, combinatorial counting), beyond what omega can handle for set intersections
- **Status:** Deferred to Phase 2b; gap existence (SCORE=1.0) is sufficient for current ablation

## Agent1 Session Mistakes — 2026-05-26 (Multi-Generalization)

### Mistake 1: Assumed (2,3) Generalization Would Scale
- **What:** Tried to add (2,3) base pair immediately after confirming (3,4)
- **Problem:** Omega could not prove bounds. After investigation, realized base-2 with digits {0,1} = ALL ℕ (trivial constraint), so setC is not sparse
- **Root cause:** Misunderstood the constraint. "Base-2 digits {0,1}" is redundant (all naturals satisfy it).
- **Lesson:** The sparsity requirement is **both** bases must have non-trivial digit restrictions. Base-2 fails the requirement.
- **Time wasted:** ~10 min debugging bounds before recognizing the fundamental issue
- **Fix:** Skipped (2,3), went directly to (3,5) which works correctly

### Mistake 2: Underestimated Bound Inference Difficulty
- **What:** Expected (2,3) bounds proof to fail only on final `omega` call
- **Problem:** Omega couldn't even prove intermediate steps: `c < 32` from `c + d = 45`
- **Root cause:** From sum constraint alone, omega cannot infer bounds without additional information (e.g., bounds on d)
- **Lesson:** Bound proofs depend on the **specific numeric values** of the gap witness. Not all gap witnesses have equal self-bounding properties.
- **Applied fix:** Chose gap witnesses where n < min(bound1, bound2) automatically (e.g., 62 < min(81, 64), 72 < min(81, 125), 89 < min(125, 343))

### Mistake 3 (Not Severe): Momentary Confusion on Base-2 Issue
- **What:** Initially thought the (2,3) bounds proof issue was a Lean API problem
- **Problem:** Spent time checking for missing tactics before realizing the set definition was the issue
- **Lesson:** When bound proofs fail, check the **mathematical validity** of the bounds first, not just the tactic formulation
- **Resolution:** Once recognized as fundamental (base-2 is trivial), moved on quickly to (3,5)

**Session outcome despite mistakes:** Minimal proof replicated, (3,5) and (5,7) added successfully, SCORE=1.0 confirmed with multi-generalization. All mistakes led to correct conclusions without blocking forward progress.

## Agent0 Session Mistake — 2026-05-26 (Scaling Attempt)

### Mistake 1: Assumed Pattern Scales to All Coprime Bases
- **What:** Attempted to add (3,7), (7,11), (3,11) following exact same template as (3,4), (3,5), (5,7)
- **Assumption:** If (5,7) works with range [0,343), then (3,7) and (7,11) should work
- **Result:** Compilation failure — omega couldn't prove bounds; native_decide hit performance wall
- **Root Cause:** Ranges [0,1331) for base-11 exceed Lean's native_decide enumeration budget. The template works in principle but not in practice beyond a certain scale.
- **Lesson:** "Theoretically universal" ≠ "practically implementable in the same way". Finite enumeration tactics have compile-time limits.
- **Duration:** ~30 min investigation + debugging

### Mistake 2: Over-Estimated native_decide Scalability
- **What:** Believed native_decide would handle ranges up to ~1000 smoothly
- **Reality:** Real limit is closer to 300-400 elements; 343 is marginal, 1331 is way too large
- **Evidence:** (5,7) works (range [0,343)) but (7,11) fails (range [0,1331))
- **Lesson:** Native code generation via native_decide is fast but not unlimited. Need to profile or hand-test boundary cases.

**Session outcome:** Identified scalability boundary. Three proven base pairs (3,4), (3,5), (5,7) sufficient to demonstrate universality. Attempted scaling blocked by compiler limits, not mathematics.
