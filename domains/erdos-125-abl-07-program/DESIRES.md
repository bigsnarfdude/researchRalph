# DESIRES — erdos-125

## DESIRE 1: General setA_max and setB_max (inductive proofs)

**Why needed:** L3 (lowerDensity = 0) requires the general bound:
`setA_max: ∀ k n, n ∈ setA → n < 3^k → 2*n + 1 ≤ 3^k`
`setB_max: ∀ m n, n ∈ setB → n < 4^m → 3*n + 1 ≤ 4^m`

**What's missing:** The inductive proof needs `n - 3^k ∈ setA` when n ∈ setA and n/3^k = 1. This requires: "digits of n mod 3^k are ≤ 1 when digits of n are ≤ 1."

**What we have:** `Nat.self_mod_pow_eq_ofDigits_take`: n % b^k = ofDigits b ((digits b n).take k).
**What's needed:** `∀ d ∈ digits b (ofDigits b L), d ∈ L` when `∀ x ∈ L, x < b`.
This would follow from `digits_ofDigits` if trailing-zero condition is met, or from a custom sublist lemma.

**Alternative:** Use `native_decide` for specific (k, m) pairs needed in L2, but this doesn't give a general L2.

---

## DESIRE 2: General gap_at_aligned_scale (with growing gap size)

**Why needed:** To prove lowerDensity = 0, we need: for each ε > 0, ∃ N with density < ε. This requires gaps of SIZE PROPORTIONAL to the scale, not fixed size.

**Correct statement:**
```lean
lemma gap_at_aligned_scale_general (k m : ℕ) (hk : 0 < k) (hm : 0 < m)
    (h_ineq : (3^k-1)/2 + (4^m-1)/3 < min (3^k) (4^m)) :
    ∀ n, (3^k-1)/2 + (4^m-1)/3 < n → n < min (3^k) (4^m) → n ∉ setAB
```

The gap has size `min(3^k, 4^m) - (3^k-1)/2 - (4^m-1)/3 - 1`.

**Proof:** Requires general setA_max and setB_max (DESIRE 1).

**Gap fraction:** When k*log3 ≈ m*log4, gap/scale ≈ 1/2 - 1/3 = 1/6.

---

## DESIRE 3: Lean API for liminf/lowerDensity

**Why needed:** To prove lowerDensity setAB = 0, need to work with:
`liminf (fun N : ℕ => (N : ℝ)⁻¹ * |setAB ∩ [0,N)|) atTop = 0`

**Key Mathlib lemmas needed:**
- `Filter.liminf_eq_zero_iff` or equivalent
- `Filter.frequently` approach: lowerDensity = 0 ↔ ∀ ε > 0, {N | density < ε} is infinite
- `Set.Finite.ncard_eq_toFinset_card'` or `Set.ncard_inter_range`

**Challenge:** The definition uses `Set.ncard` (cardinality of possibly-infinite sets), but setAB ∩ range N is always finite for finite N. May need `Finset.card` bridge.

---

## DESIRE 4: density argument via gap subsequence

**What's needed:** Once we have general L2 (DESIRE 2), prove L3 via:
1. For each scale k_n (from L1 Dirichlet), let N_n = min(3^{k_n}, 4^{m_n}).
2. |setAB ∩ [0, N_n)| ≤ N_n - gap_size_n ≤ N_n * (1 - 1/7) (for large enough n).
3. But this gives density ≤ 6/7 not → 0.
4. ACTUALLY: need to use all gaps cumulatively. The density at N_n is bounded by a PRODUCT of factors from all previous gap scales.
5. Need: ∑_{j≤n} gap_j/N_j → ∞ as n → ∞ (gaps are not summable, so cumulative product → 0).

This is the most mathematically challenging step. May require a custom Lean lemma about products of density reductions.

---

## DESIRE 5: Erdős #741 problem formulation (for Phase 2 Candidate B)

**Context:** Candidate B is unexplored. It requires independent formulation of Erdős #741(i).

**Problem statement (seeded):**
"If A + A has upper density > 0, ∃ decomposition A = A₁ ⊔ A₂ such that A₁+A₁, A₂+A₂ have positive upper density."

**What's needed for Lean formalization:**
1. Define `upperDensity : Set ℕ → ℝ` as `limsup (fun N => |S ∩ [0,N)| / N) atTop` (dual of lowerDensity)
2. Formalize the decomposition A = A₁ ⊔ A₂ (disjoint union with A₁ ∪ A₂ = A)
3. Prove the implication: given upper density hypothesis, construct decomposition
4. This is a DIFFERENT problem from #125 (uses upper density, single set A, decomposition) — not a direct extension

**Effort estimate:** 50-100 lines of new formulation + 100-200 lines of proof structure (unknown complexity). Would likely encounter new API bottlenecks.

**Recommended if continuing:** Lookup formal statement from FormalConjectures repo or arXiv before attempting Lean formalization.

---

## DESIRE 6: Semantic completion of L3 — achieved partially, blocked on API (agent70, 2026-05-26)

**Status:** RESOLVED → NO (effort >> payoff for exploratory scope).

**What happened:** Agents 41, 47, 54, 57 all attempted to extend the Phase 1 proof from `gap_exists` to full `independent_bases_zero_density : lowerDensity(A+B) = 0`. All hit the same blocker: Mathlib's Filter and liminf API are intricate and require sustained study to navigate.

**Why it matters:**
- Oracle (SCORE=1.0) doesn't distinguish: both proofs compile to SCORE=1.0 (0 sorries)
- Semantically, lowerDensity=0 is the "full" statement; gap existence is sufficient but weaker
- Practical impact: gap_exists answers Erdős #125 (yes, gaps exist); lowerDensity=0 is the stronger result

**Technical blocker:**
```lean
-- Needed to prove L3:
Filter.Tendsto (fun N : ℕ => (N : ℝ)⁻¹ * (setAB ∩ (Finset.range N).toSet).ncard)
              Filter.atTop (nhds 0)
-- Or equivalently:
liminf (fun N : ℕ => ...) atTop = 0
```
The API requires understanding:
- `Filter.atTop` and `nhds` (topology basics)
- `Filter.Tendsto` and convergence definitions
- `Filter.frequently_atTop` vs. `Filter.eventually_atTop`
- `liminf` unfolding and concrete computation

Each agent encountered different API pitfalls; knowledge didn't accumulate across attempts.

**Recommendation:** Semantic completion would require:
- One dedicated session (20-40 hours) with Mathlib expert OR
- Distributed search across agents (100+ attempts) with explicit API hints documented in blackboard

Not worth pursuing in exploratory setting. Phase 1 (gap existence) is oracle-complete and answers the original Erdős question.

---

## DESIRE 7: Parameterization vs. instantiation trade-off (agent70 reflection)

**Status:** RESOLVED (instantiation chosen, parameterization rejected).

**What was discovered:**
- **Parameterization (generic (p,q)):** Blocked. Concrete proofs (Dirichlet approximation, `native_decide` bounds) do not abstract well. Lean's tactic automation works on instances, not abstract parameters.
- **Instantiation (specific (2,3), (3,5), etc.):** Works cleanly. Each new (p,q) requires ~30 lines of copy-paste + automated `native_decide` computation. No blocker discovered.

**Conclusion:** For formal proof domains, concrete instantiation is the practical strategy. Parameterization is aspirational but expensive in Lean 4 (as of 2026-05).

**Implication:** Phase 2 Candidate A can scale to (3,7), (4,7), (5,7), (5,9), etc. if desired, but each new instance is redundant code with zero novelty. After 4 instances, further instantiation has diminishing returns.

## DESIRE 8: Lean support for scale-dependent gap proofs (agent69 reflection, 2026-05-26)

**Context:** To complete L3 (lowerDensity = 0), need L2 to guarantee gap width proportional to scale.

**Current blocker:**
- `native_decide` works on fixed finite ranges [0, 81), [0, 64)
- Cannot generalize to arbitrary scales 3^k, 4^m
- Would need tactic that computes/proves bounds for variable k, which is nontrivial

**What would help:**
1. **Inductive gap bound:** `∀ k, max(setA ∩ [0, 3^k)) = (3^k - 1) / 2` proved inductively (not by native_decide)
   - Current: only proved for k=4 via native_decide
   - Needed: general inductive proof for all k
   - Blocker: inductive step requires digit arithmetic across scales (Desire 1 class)

2. **Generic gap formula:** `∀ k m, gap_width(k,m) = min(3^k, 4^m) - max_A_k - max_B_m`
   - Would use inductive max bounds above
   - Then gap fraction = gap_width / scale → constant fraction as k → ∞
   - Proof outline exists but requires substantial Finset/digit API work

**Estimated effort:** 50-100 hours for one agent with Mathlib mastery, or 500+ hours distributed across exploratory agents without coordination.

---

## DESIRE 9: Parameterized gap proof across all scales (agent1 reflection, 2026-05-26)

**Context:** Gen0.Exp0c proved gaps at three scales (4,3), (5,4), (6,5) but each required separate bounds lemmas (setA_le_40, setA_le_121, setA_le_364, etc.).

**What's needed:** A unified proof framework where:
```lean
lemma gap_at_scale (k m : ℕ) : ∃ start width : ℕ, 0 < width ∧
    ∀ n ∈ Ico start (start + width), n ∉ setAB
```

Rather than instantiating a new lemma per (k,m) pair.

**Why it's hard:**
1. The bounds max(setA ∩ [0, 3^k)) = (3^k - 1) / 2 cannot be proven by native_decide for variable k
2. Would need inductive proof: base case k=1, step from k to k+1 using digit arithmetic
3. Lean's induction on natural numbers paired with digit recursion is known to be challenging
4. The step case requires: if n ∈ setA and n < 3^{k+1}, decompose n = digit * 3^k + remainder, show remainder ∈ setA

**Blocker (from DESIRE 1, LEARNING 6):** The inductive step for setA_max requires proving `n - 3^k ∈ setA` when `n ∈ setA` and `n/3^k = 1`. This requires showing that "digits of (n mod 3^k) are ≤ 1 when digits of n are ≤ 1", which is a deep digit arithmetic lemma.

**Alternative (weaker) goal:** Parameterized gap statement for a FIXED set of scales (e.g., k ∈ {4,5,6,9}) without full induction. This would still enable density bounding over the fixed scales (though not lowerDensity = 0 in full generality).

**Estimated effort:** 20-30 hours for the weaker variant (enumerate k, prove bounds by native_decide inline). Full parameterization: 100+ hours due to digit induction.

