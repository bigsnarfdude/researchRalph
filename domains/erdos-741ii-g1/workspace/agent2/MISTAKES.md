# Agent2 Mistakes

## Attempt 1: Using Singleton Set Notation `{ck k}`

**What**: Tried defining stagek as `fun k => {ck k} ∪ Bk k ∪ Fk k`

**Result**: "Function expected at" parse errors throughout

**Lesson**: In Lean 4, `{ck k}` doesn't create a singleton set in all contexts. Need explicit set constructor syntax or use `insert` or `Set.insert`.

## Attempt 2: Recursive Definition with `fun k => match k with`

**What**: Tried `def Akn : ℕ → Set ℕ := fun k => match k with | 0 => ... | k+1 => Akn k ∪ ...`

**Result**: Recursive reference to Akn in the match branches caused "Function expected" errors.

**Lesson**: Pattern-match-style definitions with recursion need the bare `def Akn : Type | 0 => ... | k+1 => ...` syntax, not the `fun k => match` form.

## Attempt 3: Direct `Or.inl`/`Or.inr` on Nested Disjunctions

**What**: After proving `hx : x = 2 ∨ x = 3`, tried `exact Or.inl hx` to show `x = 2 ∨ x = 3 ∨ ∃ k, ...`

**Result**: "Application type mismatch" - type checker couldn't see that `P ∨ Q → (P ∨ Q) ∨ R`

**Lesson**: Nested disjunctions `P ∨ Q ∨ R` associate as `P ∨ (Q ∨ R)`, so simple `Or.inl` doesn't work. Need either:
- Explicit case-by-case construction with multiple `left`/`right` tactics
- Define setA differently (e.g., as union or explicit predicate)
- Or use `⟨proof, ...⟩` notation which seems to work better (but I didn't fully test)

## Attempt 4: Using `left; exact h` Then `right; left; exact h`

**What**: In cases on `hx : x = 2 ∨ x = 3`, did `left` to focus on first disjunct, then `exact h`

**Result**: "Type mismatch" - `left` tactic on a triple disjunction doesn't work as expected

**Lesson**: Multi-goal tactics like `left`/`right` are for binary Or only. Need to be explicit about which constructor to apply.

## Attempt 5: Induction on k With Direct Case Rules

**What**: `induction k with | zero => ... | succ k ih => intro x hx; ...`

**Result**: Some cases worked, but others had goal structure mismatch because intro pattern didn't align with hypothesis flow

**Lesson**: In Lean 4, after induction, need to be careful about when to intro variables. Better to intro first, then induction.
