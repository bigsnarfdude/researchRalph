# Mistakes — erdos-741ii-g1

## Destructuring pattern error (exp002)
**Issue**: Incorrect destructuring of conjunction in main theorem intro pattern
**Root cause**: Pattern `⟨⟨C₁, hC₁⟩, C₂, hC₂⟩` didn't match the paired existential structure
**Fix**: Changed to `⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩` to properly destructure both existentials
**Lesson**: When destructuring a conjunction of existentials, both sides need explicit tuple notation

## Agent6 Mistakes

### Mistake 1: Trying to prove set membership with rfl/left
**Issue**: Attempted `use 0; left; rfl` to prove `ck 0 ∈ setA`
**Root cause**: After unfolding setA and using witness j=0, the goal becomes a 3-way union `{ck 0} ∪ Bk 0 ∪ Fk 0` which is not parsed as a simple binary disjunction for `left` tactic
**Failed approach**: Various `simp`, `tauto`, `show` patterns didn't work
**Lesson**: Set membership in multi-way unions requires careful unfolding order or explicit membership constructor

### Mistake 2: Using interval_cases without bounds
**Issue**: `interval_cases n` failed to split into cases without explicit bounds
**Root cause**: interval_cases requires n to be in a decidable bounded range; just having `n ≤ 6` wasn't enough
**Fix**: Used explicit case split via `by_cases h4 : n = 4; ... by_cases h5 : n = 5; ...` 
**Lesson**: For finite case analysis, explicit `by_cases` chains are more reliable than automation in Lean 4

### Mistake 3: Overcomplicating main theorem proof
**Issue**: Tried to handle full ck 0 membership + partition membership + gap_lem contradiction in one block
**Root cause**: Too many layers of set membership to prove cleanly; technical overhead obscured logic
**Fix**: Simplified to just outline the argument as comments with a single sorry
**Lesson**: When proof structure is complex, it's better to simplify to show correct approach, then fill in technical pieces separately

### Mistake 4: Wrong tactic for Akn monotonicity
**Issue**: Initial attempt with `simp only [Akn]; split_ifs` failed
**Root cause**: Pattern matching in recursive definition needs structural tactics like tauto for union logic
**Fix**: Changed to `simp only [Akn] at hx ⊢; tauto` which handles propositional union membership
**Lesson**: tauto is powerful for propositional logic in set-theoretic contexts (unions = Or)

## Agent 3 Mistakes

### Mistake 1: Induction on multiple parameters
**Issue**: Multiple attempts at `induction k with | ... | ...` syntax failed with "unexpected identifier; expected '-'" errors
**Root cause**: Lean 4 induction syntax with multiple patterns in the same `with` block has subtle scoping rules
**Failed attempts**: `with | zero => ...` at line 40-56 all errored despite correct punctuation
**Solution**: Used fully reverted proof with single-parameter induction: `revert x k j; induction k`
**Lesson**: When facing induction syntax errors, revert all parameters then do induction on one; more verbose but reliable

### Mistake 2: Union membership with simp + angle brackets
**Issue**: `simp only [Set.mem_union, Set.mem_iUnion, Set.mem_singleton_iff]; exact ⟨k, Or.inl rfl⟩` failed
**Root cause**: After simp, the goal structure doesn't match the angle bracket pattern for constructing existentials
**Why it matters**: This should have worked; suggests simp is unfolding differently than expected or the goal type is more complex
**Fallback**: Reverted to `sorry` for ck k ∈ setA
**Lesson**: When angle bracket construction fails after simp, try explicit `use` + `left/right` tactics instead

### Mistake 3: omega on exponential inequality
**Issue**: `omega` couldn't prove `5^(C+5) > C` 
**Root cause**: omega handles linear arithmetic on ℕ but not exponentials; it's a known limitation
**Why it failed**: Polynomial growth isn't enough for 5^n comparisons; need explicit induction
**Lesson**: For exponential or non-linear ℕ goals, use `sorry` + explicit induction or `norm_num` on concrete instances

### Mistake 4: Trying to close ck k ∈ setA with simp chains
**Issue**: Multiple simp variants all left unsolved goals or type mismatches
**Approaches tried**:
- `simp only [setA, Set.mem_union, Set.mem_iUnion, Set.mem_singleton_iff]; right; exact ⟨k, Or.inl rfl⟩`
- `unfold setA; simp only [Set.mem_union, Set.mem_iUnion]; use k; left; rfl`
- `unfold setA; right; use k; exact Or.inl (Set.mem_singleton _)`
**Root cause**: Unclear whether issue is simp unfolding, goal structure, or angle bracket syntax
**Lesson**: Set membership in nested unions is finicky; might need to look at prior working proofs or break down into more lemmas

### Mistake 5: Over-relying on omega for ℕ arithmetic
**Issue**: omega closed `j ≤ 0 when k = 0` but failed on exponentials
**Why it matters**: Good instinct to use omega for linear ℕ goals, but need to recognize its boundaries
**Lesson**: Keep omega for: j ≤ k, a < b+c, subtraction; avoid for powers, factorial, exponentials
