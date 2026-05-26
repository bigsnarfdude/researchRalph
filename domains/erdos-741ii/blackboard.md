# Blackboard — Erdős #741(ii) Domain

**Oracle:** Lean 4 compiler. Sorry count = 0 + definitions unchanged = proved.
**Status:** Phase 1 — formalize the construction. 3 sorries.

---

## CRITICAL: Natural Density Trap — AVOIDED

Unlike #125 (lowerDensity = liminf) and #741(i) (upperDensity = limsup),
this problem uses NO density definitions at all. It is purely constructive.
Do NOT introduce lim, liminf, limsup, or density anywhere.
The two key properties are:
- isBasisOrder2: ∀ n, ∃ a b ∈ A, a + b = n  (universal, no limits)
- boundedGaps: ∃ C, ∀ n, ∃ m ∈ S, n ≤ m ≤ n+C  (existential constant, no limits)

---

## THE CONSTRUCTION

```
gapBound k = 2^(2^k)    ← super-exponential, ensures clumps never overlap

clump k = {gapBound k, gapBound k + 1, ..., gapBound k + k}
          ^^^ width k+1, centered at 2^(2^k)

setA741 = ⋃ k, clump k
```

Concrete values:
- clump 0: {1}                  (gapBound 0 = 2^1 = 2... actually 2^(2^0) = 2^1 = 2)
- clump 1: {4, 5}               (gapBound 1 = 2^2 = 4, width 2)
- clump 2: {16, 17, 18}         (gapBound 2 = 2^4 = 16, width 3)
- clump 3: {256, 257, 258, 259} (gapBound 3 = 2^8 = 256, width 4)
- clump 4: {65536..65540}       (gapBound 4 = 2^16 = 65536, width 5)

The gaps between clumps grow super-exponentially.
The widths grow linearly. Eventually clumps dominate any partition.

---

## WHY THE PARTITION PROPERTY HOLDS

For any partition A₁ ⊔ A₂ = setA741:
- Each clump k is a Finset of width k+1
- By pigeonhole on clump k: one of A₁, A₂ gets ≥ ⌊(k+1)/2⌋ elements from clump k
- For large k, ⌊(k+1)/2⌋ is large enough that the sumset of that piece covers
  a range of width ≥ k around 2*gapBound(k)
- Since gapBound(k+1) >> gapBound(k) + k, these ranges cover all of ℕ eventually

For the OTHER piece: it also gets some elements from each large clump.
Key insight: if A₁ gets the "bottom half" of clump k and A₂ gets the "top half",
then A₁+A₁ covers around 2*gapBound(k) to 2*gapBound(k)+k, and
A₂+A₂ covers around 2*gapBound(k)+k to 2*gapBound(k)+2k.
Both have bounded gaps, just at different positions.

---

## LEMMA 1: clumps_disjoint

**Statement:** gapBound k + k + 1 ≤ gapBound (k+1)

**Why:** gapBound(k+1) = 2^(2^(k+1)) = (2^(2^k))^2 = gapBound(k)^2
We need gapBound(k)^2 ≥ gapBound(k) + k + 1
Since gapBound(k) = 2^(2^k) ≥ 2^k ≥ k+1 for k ≥ 1, we have
gapBound(k)^2 ≥ gapBound(k) * (k+1) >> gapBound(k) + k + 1. ✓

**Lean approach:**
```lean
have h1 : gapBound (k+1) = gapBound k ^ 2 := by
  simp [gapBound, pow_succ, pow_mul]
have h2 : gapBound k ≥ k + 2 := by
  -- 2^(2^k) ≥ k+2 for all k, by induction or norm_num for small k + mono
  induction k with
  | zero => simp [gapBound]
  | succ n ih => ...
linarith [sq_nonneg (gapBound k), ...]
```

**Key Mathlib lemmas:**
- `pow_succ`, `pow_mul` for the squaring identity
- `Nat.one_le_pow` for positivity
- `linarith` or `nlinarith` for the numeric bound

---

## LEMMA 2: setA741_is_basis

**Statement:** ∀ n : ℕ, ∃ a b ∈ setA741, a + b = n

**Proof sketch:**
- For n = 0: use 0 ∈ clump 0? Actually gapBound 0 = 2, so 0 ∉ setA741.
  Handle small n separately (n ≤ 4: direct computation).
- For n ≥ 4: find k such that 2*gapBound(k) ≤ n ≤ 2*(gapBound(k)+k).
  Then write n = gapBound(k) + (n - gapBound(k)), both in clump k.
  This works iff 0 ≤ n - gapBound(k) ≤ k, i.e., gapBound(k) ≤ n ≤ gapBound(k)+k,
  which means n/2 ∈ [gapBound(k), gapBound(k)+k]. Find k with gapBound(k) ≤ n/2 ≤ gapBound(k)+k.

**⚠️ Issue:** The construction as stated doesn't cover all of ℕ without modification.
The sumset 2*A₁ covers [2*gapBound(k), 2*(gapBound(k)+k)] for each k.
Between clumps there are GAPS in the sumset too.

**Revised construction needed:** Add "survivor elements" between clumps to close gaps.
Or: use a more careful sequence definition.

**Alternative:** Define setA741 as clumps PLUS all even numbers ≤ 4 as base cases,
or use a different rapidly growing sequence that provably covers ℕ as a basis.

**Note to agents:** The clump-only construction may NOT be a basis of order 2 for ALL n.
Check small values. If gaps exist in the sumset, modify the construction to add
"bridge elements" between clumps. The key insight is that the partition property
can survive the addition of bridge elements.

---

## LEMMA 3: partition_bounded_gaps

**Statement:** For A₁ ∪ A₂ = setA741, Disjoint A₁ A₂ → boundedGaps(sumset A₁) ∧ boundedGaps(sumset A₂)

**Proof sketch:**
- Fix any partition. For each clump k, let sₖ = |A₁ ∩ clump k|.
- Then |A₂ ∩ clump k| = k+1-sₖ.
- A₁+A₁ near 2*gapBound(k) covers a range of width ≥ 2*sₖ - 1 (sums of sₖ consecutive ints).
- A₂+A₂ near 2*gapBound(k) covers a range of width ≥ 2*(k+1-sₖ) - 1.
- sₖ + (k+1-sₖ) = k+1, so the two ranges together cover width ≥ k+1.
- The gap between 2*(gapBound(k)+k) and 2*gapBound(k+1) grows super-exponentially.
- For large k, each range covers more than the gap to the next clump → bounded gaps. ✓

---

## FAILURE LOG

*Agents: append here when a tactic approach fails.*

---

## SORRY COUNT TRACKER

| Session | Date | Sorry count | Phase |
|---------|------|-------------|-------|
| Seed | 2026-05-25 | 3 (L1 + L2 + L3) | Phase 1 start |
