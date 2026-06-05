# Agent 12 Desires — What Would Help Complete This Proof

## Critical Missing Lemmas

### 1. Stage Bounds Lemma
```lean
lemma stage_bounds (k j : ℕ) (hj : j < k) :
    -- All elements from stage j are at most some function of Q j
    (∀ x ∈ Bk j, x ≤ 6 * Q j) ∧
    (∀ x ∈ Fk j, x ≤ 15 * Q j) ∧
    (ck j = 4 * Q j)
```
**Why needed**: Would let me bound sum of elements from stage j' < k against Jk k = [9*Q k, 10*Q k)

### 2. Coverage Lemma  
```lean
lemma coverage_by_stages (k : ℕ) (x : ℕ) (hx : x ∈ Icc 4 (6 * Q k)) :
    ∃ j < k, x ∈ Bk j ∪ Fk j ∪ {ck j}
```
**Why needed**: Would immediately complete basis_lem by showing x - 2 falls in some stage

### 3. Rigidity Helper
```lean
lemma rigidity_by_decomposition (k : ℕ) (x : ℕ) (hx : x ∈ Jk k) :
    ∀ a b : ℕ, a + b = x → a ∈ Akn (k+1) → b ∈ Akn (k+1) →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k)
```
**Why needed**: A helper that encapsulates the stage decomposition logic, breaking down rigidity into bite-sized pieces

## Existing Mathlib Gaps

### ℕ Subtraction Support
`omega` doesn't handle ℕ subtraction well in compound goals. Would help to have:
- `Nat.sub_le_iff_le_add` — more versions of subtraction reasoning
- Better automation for `nat_sub_trichotomy` style reasoning

### Set Theory Automation
No good tactic for "derive False from x ∈ S ∩ T when S ∩ T = ∅".
- Currently resort to `rw [empty_eq] at h; simp at h`
- Could use a `set_disjoint_absurd` tactic

## Capability Improvements Needed

### Better Case Analysis Tools
The rigidity lemma has ~8+ cases across different stage pairs. A case automation tactic that tracks which stages have been covered would help.

### Interval Arithmetic
Many proofs boil down to "a ≤ b and b < c implies a < c". Would be nice to have:
- `interval_trans: interval_le : lemma_tactics for [x, y] arithmetic
- Automatic bound propagation in `omega`

## Workarounds That Worked

1. **Explicit stage lemmas**: Rather than abstract over all stages, write out Akn(k+1) = {2,3} ∪ ⋃ j<k+1, ... explicitly
2. **Use induction for exponential claims**: Base + step for "Q k grows" claims worked better than direct polynomial arguments
3. **Construct witnesses explicitly**: Rather than assert existence, use `use j; ⟨proof, proof⟩` pattern

## Documentation Improvements

Would help to have:
- Explicit size lemmas for Bk, Fk, ck early on: `lemma Bk_size`, `lemma Fk_size`
- A "stage anatomy" lemma showing how levels nest: levels grow exponentially, later stages dominate
- Examples of decompositions at specific k values (k=1, k=2) to build intuition

## Estimated Effort to Complete

- **Stage bounds lemma**: ~30 lines, straightforward unfold + omega
- **Coverage lemma**: ~50 lines, requires by_cases on which region x falls in
- **Rigidity decomposition**: ~150 lines, full case analysis but now compartmentalized
- **Total**: ~3-4 hours of focused work to close all 3 sorries

The proof structure is sound; completing it is primarily a matter of developing these supporting lemmas.
