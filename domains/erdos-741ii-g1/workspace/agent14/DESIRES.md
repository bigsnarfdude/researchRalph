# Agent 14 Desires — Erdős #741(ii)

## Capabilities Needed

### 1. Set Membership Tactics
- A tactic that automatically resolves membership in unions and set literals
- Current: must manually construct `Or.inr (Or.inr ...)` chains
- Desired: `membership_auto` that figures out the right branch and constructs the proof

### 2. Interval Arithmetic
- Better support for nat subtraction in interval proofs
- Current: `omega` works but requires explicit bounds
- Desired: tactic that automatically verifies `x ∈ Icc a (b - 1)` from `a ≤ x ≤ b - 1`

### 3. Multi-Stage Induction
- Helper lemma for "every n ≥ k belongs to some Akn m"
- Current: must prove manually for each m
- Desired: automated staging lemma builder

### 4. Constructor Synthesis
- When goal is `∃ a b, P a b n`, synthesize plausible witnesses
- Current: must manually identify witness pairs
- Desired: tactic that searches setA for pairs summing to n

### 5. Documentation
- Clearer error messages when `right` fails (explain binary vs. n-ary structure)
- Examples in Mathlib for handling 3+ branches of unions
- Lean 4 set notation guide with membership proof patterns

## Architectural Improvements Needed

### 1. Basis Lemma Strategy
- Current approach (prove all n directly) is tedious
- Better: split into stages—prove that each level k covers an interval
- Then combine: `n ∈ [4, Q k]` → `n ∈ Akn k + Akn k`

### 2. Rigidity Lemma Structure  
- Prove stage-by-stage: elements from stage j < k are too small, j > k too large
- Bundle this into lemmas about stage decomposition
- Use these to close rigidity

### 3. Gap Lemma Dependencies
- Currently requires full rigidity proof
- Could weaken: only need "if ck k ∉ T, then Jk k is sparsely hit by T+T"
- This might be provable more directly from geometry

## Test/Verification Features

1. **Automated SORRY counting:** Track which proofs are which `sorry` for prioritization
2. **Partial score:** Points for number of sorries eliminated (not just SCORE=1.0 or 0)
3. **Proof sketches:** Allow `sorry` with reason string, extract to feedback
4. **Lemma dependency graph:** Visualize which lemmas block which proofs
