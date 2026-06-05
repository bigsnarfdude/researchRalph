# Agent 8 Mistakes — Erdős #741(ii) Implementation

## Failed Approaches

### 1. Unfold on Recursive Definitions
**What**: Tried `unfold Akn` on the recursive definition `Akn k | 0 => ... | k+1 => ...`
**Result**: Type mismatch errors; `simp only [Akn]` didn't unfold properly
**Lesson**: Use `simp only [Akn, Set.mem_union]` combined with `tauto` instead of manual unfold

### 2. `simp [ck, Q]` for Membership in Set Literals
**What**: Tried `simp [ck, Q]` to prove 4 ∈ {ck 0}
**Result**: "Tactic `simp` made no progress" 
**Issue**: Set literal membership requires membership lemmas, not just unfold
**Fix**: Use `Set.mem_singleton_iff` explicitly

### 3. Using `show` After `right` in Multi-Way Union
**What**: After `right`, used `show ∃ x, 4 ∈ ⋃ x, ...`
**Result**: "'show' tactic failed, pattern does not match"
**Issue**: The goal after `right` has a different structure than expected
**Fix**: Use `use 0` directly after `right`, let Lean infer types

### 4. `left` Tactic on Three-Way Union
**What**: Used `left` on `{ck 0} ∪ Bk 0 ∪ Fk 0`
**Result**: "left tactic works for inductive types with exactly 2 constructors"
**Issue**: Multi-way unions parse as nested binary unions, but structure unclear
**Fix**: Use `simp only [Set.mem_union, ...]` to unfold, then `norm_num`

### 5. Direct `decide` for Set Membership
**What**: Used `decide` to prove membership in setA
**Result**: "Tactic `decide` failed for proposition"
**Issue**: `decide` only works for decidable propositions; set membership with infinite unions not decidable
**Fix**: Manual proof or `sorry` (set membership proofs remain incomplete)

### 6. Trying to Prove `akn_mono` via Case Analysis
**What**: Attempted `cases k with | zero => ..., | succ k => ...`
**Result**: Type mismatch on both branches
**Issue**: Recursive definition of Akn doesn't unify with case goals
**Fix**: Direct proof using `simp only [Akn, Set.mem_union]; tauto`

### 7. `unfold Set.mem_add` in `basis_lem`
**What**: Used `unfold Set.mem_add` to unfold additive set membership
**Result**: "Tactic `unfold` failed to unfold `mem_add`"
**Issue**: `mem_add` is not a definition that unfolds; it's a lemma
**Fix**: Remove the unfold, let `sorry` handle the goal

### 8. Over-Zealous `simp` on Membership Proofs
**What**: Used `simp only [...]` with too many lemmas on hck_exists
**Result**: "unsolved goals" after simp but no visible remaining goal
**Issue**: `simp` was partially solving and leaving malformed goals
**Fix**: Simpler proof structure or `sorry`

### 9. Using `rfl` After `simp` on Q_succ
**What**: After `simp only [pow_succ]`, used `rfl` to close goal
**Result**: "The left-hand side ... is not definitionally equal to ..."
**Issue**: `simp` produces a form that's not exactly `5 * (5^k)` 
**Fix**: Use `simp [Q, pow_succ, mul_comm]` which handles rewriting fully

## Pattern Recognition

### What Worked Well
- **`tauto` on union structures**: Correctly handles logical tautologies in set membership
- **`simp [lemma_names]`**: Most powerful when lemma names explicitly guide rewriting
- **`omega` for ℕ subtraction**: Essential for Bk, Fk definitions with Nat.sub
- **`norm_num` for ground arithmetic**: Works on concrete numeric goals
- **`by_cases` for decomposition**: Useful for stage analysis in rigidity

### What Didn't Work
- Manual recursion handling on Akn
- Expecting `show` to preserve goal structure after case splits
- Trying to decide infinite set membership
- Over-relying on `simp` without explicit guidance

## Meta-Lesson

Lean 4 tactic-based proofs require understanding the exact syntactic form of goals. Many failures were due to:
1. Goal form after tactic not matching what I wrote
2. Union notation parsing as nested binary unions
3. Membership notations not unfolding as expected

The solution is typically to:
- Check goal form with `exact?` or `sorry` + hover
- Break complex goals into smaller pieces
- Use explicit lemma names rather than aggressive automation
