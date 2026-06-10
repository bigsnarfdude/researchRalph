# LEARNINGS — agent9 (erdos-741ii cold)

## Problem structure
- Part 1 (basis): A+A ⊇ [4,∞). Part 2: ∀ 2-partition A=A₁⊔A₂, ¬(both A_i+A_i syndetic).
- IsSyndetic S ⟺ bounded gaps. Subset of syndetic CAN be non-syndetic.

## Hard constraints derived
1. If A contains ANY syndetic set, a residue/parity split makes both sumsets syndetic
   ⇒ part 2 false. So A MUST be a thin basis (density → 0), using large+large sums.
2. A single digit-closed basis with A+A = ℕ and k·A ⊆ A dies to the lowest-digit coloring
   (A₁ = base·A, A₂ = c + base·A; both sumsets = arithmetic progressions, syndetic).
   ⇒ the construction must be a UNION of incompatible structures to break shift-closure.
3. Combination: thin (needed for truth) ⟹ basis proof is non-trivial (digit decomposition);
   the partition property is a research-level structural (non-counting) theorem. Pure
   density counting is too weak: |A∩[0,N]| ~ 2√N admits both halves clearing the √(N/C)
   counting floor, so no contradiction from counting alone.

## Lean / Mathlib facts that worked
- `Nat.ofDigits_cons : Nat.ofDigits b (d :: L) = d + b * Nat.ofDigits b L`
- `Nat.ofDigits_digits : Nat.ofDigits b (Nat.digits b n) = n`
- `Nat.digits_lt_base (by norm_num) hmem : d < b` for d a digit.
- Digit-split basis proofs: define A = {ofDigits b L : ∀d∈L, d≤bound}; build the two
  summands as `ofDigits b (digits b n |>.map split_fn)`; prove the sum by list induction
  where the inductive step closes with `omega` (it natively handles `%` and `/` by literals
  and the linear combination with the IH atoms).
- Unfold set membership for omega/exact with `simp only [Set.mem_setOf_eq]` or `show`.

## Best construction found
A = E ∪ O (binary even/odd bit positions) = {base-4 {0,1}-digit numbers} ∪ {2·same}.
Basis FULLY PROVED in Lean. Survives all simple coloring attacks. Part 2 = open crux.
</content>
