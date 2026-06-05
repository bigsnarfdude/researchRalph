# Agent13 Mistakes & Dead Ends

## Critical Blocker: akn_mono Type Error

Attempted 5+ approaches to prove `akn_mono : ∀ k, Akn k ⊆ Akn (k + 1)`:

1. **Direct tactic proof**:
```lean
intro k x hx
unfold Akn at hx ⊢
exact Or.inl hx
```
Result: "Application type mismatch: The argument" at line with `Or.inl hx`

2. **Using Set.mem_union_left**:
```lean
fun k x hx => Set.mem_union_left _ hx
```
Result: Same type mismatch

3. **Calc block approach**:
```lean
calc x ∈ Akn k := hx
  _ ⊆ Akn k ∪ {...} := ...
  _ = Akn (k + 1) := ...
```
Result: "invalid 'calc' step" — types don't match in calc chain

4. **Lambda + tactic hybrid**:
```lean
fun k => by intro x hx; unfold Akn...
```
Result: Same type error persists

**Conclusion**: The error is not in our proof strategy but in type checking. The exact same pattern works in gap_lem proof, so the issue may be specific to this lemma's signature or how the definition unfolds. Suggests a deeper Lean 4 parsing issue, not a proof logic problem.

**Workaround**: Leave as sorry; future agent can try:
- Different syntax like `Set.Subset.trans`
- Using tactic mode differently
- Checking if there's a library lemma for this

---

## basis_lem Coverage Challenge

Attempted detailed case split:

```lean
by_cases h : x ≤ 5 * Q k
· -- Case 1: x = (x - 2Q) + 2Q
  have h1 : 2 * Q k ≤ x := by omega
  use x - 2*Q k
  -- Then need: (x - 2Q) ∈ Akn(k+1) ⊆ Bk k
  simp [Akn, Bk]; omega
```

**Problem**: After simp, the goal becomes something like:
```
⊢ x - 2*Q k ≤ 6*Q k - 1 ∨ ...other cases...
```

Omega can't handle this because:
1. The goal has become a disjunction (union membership)
2. The arithmetic (nat subtraction) is mixed with disjunction structure
3. Simp opens too much; omega can't find the right pieces

**Failed fixes**:
- Using `simp only [mem_Icc]` to isolate arithmetic
- Splitting with `by_cases` on the membership itself
- Using `norm_num` for ground goals first

**Key lesson**: Don't simp goals that mix membership and arithmetic. Keep them separate or use explicit membership introduction patterns.

---

## Or Constructor Syntax Issues

Multiple attempts with nested unions like `A ∪ B ∪ C ∪ D` failed:

```lean
simp only [Set.mem_union]
right; right; left
-- Error: right tactic only works with 2 constructors
```

**Root cause**: `A ∪ B ∪ C ∪ D` parses as `A ∨ (B ∨ (C ∨ D))`, not 4 independent constructors. The `right` tactic expects exactly 2.

**Working solution**: Use `Or.inr (Or.inr (Or.inl h))`

This pattern works in gap_lem but fails in Akn_sub_setA induction. Unclear why the difference.

---

## Induction on Akn Recursion

Tried:
```lean
induction k with
| zero => ... simp [Akn]
| succ k ih => ... simp [Akn]; rcases hx with a | b | c | d
```

**Problem**: After unfolding Akn recursively, the induction hypothesis `ih` doesn't have the right type to apply to the recursive case. The membership in the new level mixes old and new level membership.

**Attempted fix**: Use `generalizing x` to apply ih correctly — still fails on or-constructor patterns.

---

## Arithmetic Bounds: n ≤ 6 * 5^n

Tried multiple approaches:

1. **Direct omega**: 
```lean
unfold Q; omega
```
Result: omega doesn't handle exponentiation in goals

2. **Induction with nlinarith**:
```lean
induction n with
| zero => norm_num
| succ n ih => nlinarith
```
Result: nlinarith can't connect exponential growth to linear bound

3. **Manual witness**:
```lean
have : 5 ^ n ≥ n := by induction...
linarith
```
Result: The inductive proof of 5^n ≥ n itself fails in succ case with arithmetic tactics

**Current status**: Left as sorry. Should work but requires careful arithmetic setup.

---

## What We'd Do Differently

1. **Avoid simp on mixed membership + arithmetic** — breaks omega
2. **Be explicit with Or patterns** — use concrete constructors, not left/right tactics
3. **Keep arithmetic goals isolated** — separate membership proof from arithmetic
4. **Trust the LEARNINGS for gap_lem** — the pattern that worked there should replicate

---

## Summary

- **1 hard blocker** (akn_mono type error — might be Lean issue)
- **3 medium blockers** (basis_lem, Akn_sub_setA, rigidity structure)
- **2 easy blockers** (bound proofs, partition body)

The core proofs are sound; execution is the bottleneck.
