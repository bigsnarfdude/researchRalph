# Agent7 Mistakes and Lessons

## Issues Encountered

### 1. Pattern Matching on Negations
**Mistake**: Tried to destructure `¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂))` directly in intro with `⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩`
**Lesson**: Negations don't destructure in intro patterns. Need to intro the negation first, then work with the contents separately via have statements or by providing the hypothesis to the negation.
**Fix**: Use `intro ... h_both` and then `obtain ⟨..., ...⟩ := h_both.1/h_both.2` for conjunction components

### 2. Set Union Constructor Issues  
**Mistake**: Used `Or.inl` directly for union membership, but Lean's set unions have a more complex structure
**Lesson**: Set membership in unions unfolds to disjunctions, but the structure depends on nested unions (left-associative)
**Fix**: For `A ∪ B ∪ C` which is `(A ∪ B) ∪ C`, membership needs `Or.inl (Or.inl hx)` for A, `Or.inl (Or.inr hx)` for B, `Or.inr hx` for C

### 3. Simp-induced Rewriting Issues
**Mistake**: Used `simp only [Akn]` which rewrote the variable in the context, making subsequent proofs fail
**Lesson**: `simp` can rewrite hypothesis types unexpectedly. `unfold` is safer for definitions
**Fix**: Use `unfold Akn at *` carefully, and verify the resulting goal/context

### 4. Interval Membership Unfolding
**Mistake**: Tried to use `norm_num` on Icc/Ico membership without proper unfolding
**Lesson**: `Set.mem_Icc`, `Set.mem_Ico`, etc. require explicit unfolding or simp lemmas
**Fix**: Use `simp only [Icc, Ico, Set.mem_singleton_iff]` to convert to inequality goals

### 5. Omega vs Linarith
**Mistake**: Used `linarith` on natural number subtraction goals (e.g., `n - m : ℕ`)
**Lesson**: `linarith` does NOT handle `ℕ` subtraction. `omega` is required for natural number arithmetic
**Fix**: Always use `omega` for ℕ goals

## What Worked Well

1. **Modular structure**: Defining Q, ck, Bk, Fk, Jk, setA, Akn upfront made the proof structure clear
2. **Gap lemma approach**: The key insight that gap_lem can use rigidity_lem to enforce the partition is sound
3. **Type-aware lemma statements**: Keeping lemma types explicit helped catch errors early

## Why the Proof Remains Incomplete

The remaining 5 sorries require:
- **rigidity_lem**: 16-case analysis (4 element classes × 4) to show interval sums have specific form
- **stage_coverage**: Explicit coverage of [4*Q(k+1), 6*Q(k+1)] via 8 pair types  
- **basis_lem**: Inductive cover of all n ≥ 4 using stage_coverage
- **elem_from_stage**: Helper lemma unfolding setA definition (union of unions)
- **Main theorem**: Contradiction argument picking k where Q k dominates gap constants

These are all feasible but require substantial boilerplate interval arithmetic proofs.
