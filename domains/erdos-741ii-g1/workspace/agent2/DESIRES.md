# Agent2 Desires

## Language/Library Features

1. **Better disjunction handling**: A tactic or lemma that automatically lifts `P ∨ Q` to `P ∨ Q ∨ R` would save hours. Currently manual case handling is error-prone.

2. **Set membership automation**: A tactic that unfolds set predicates and automatically applies constructors for existentials and disjunctions in membership goals.

3. **Clearer error messages for type mismatch in Or/And constructors**: The "Application type mismatch" errors didn't pinpoint the associativity issue.

## Context/Documentation

1. **Explicit Akn semantics**: A working example of how to define recursive set-returning functions in Lean 4 (pattern-match syntax vs. fun syntax) would have saved time.

2. **Disjunction associativity rules**: Clear documentation that `P ∨ Q ∨ R` parses as `P ∨ (Q ∨ R)` and how to handle it.

3. **Worked examples from program.md in Lean syntax**: Even one small sub-proof showing how to prove setA membership would have unblocked this.

## Tools/Tactics

1. **`finish` or `decide` tactic**: Something to automatically solve membership goals for finite predicates.

2. **Tactic to unfold + normalize disjunctions**: Turn `P ∨ Q ∨ R` into a list and match/construct systematically.

3. **Exploration mode**: Way to see the exact goal state and hypothesis types during interactive proof, not just in error messages.

## Proof Strategy

1. **Alternative setA definition**: Define setA as a union or explicit decidable predicate instead of nested disjunction - might sidestep type issues.

2. **Akn completeness**: Prove once that Akn is increasing and covers all of setA, then reuse that fact systematically.

3. **Syndetic auxiliary lemmas**: Library of lemmas about syndetic sets (gaps, gaps in complements, etc.) to avoid reproving rigor lemmas.
