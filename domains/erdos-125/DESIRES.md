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
