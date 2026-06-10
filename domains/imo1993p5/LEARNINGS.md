# Learnings — imo1993p5

## Lean 4 Pattern Matching and Definitional Equality

When a function is defined via pattern matching on constructor arguments (e.g., `| 0 => 0 | n+1 => ...`), the resulting definition is **not strictly definitional** for proving lemmas via `rfl`. Instead:
- Use `simp [function_name]` to unfold and simplify
- Use `norm_num [...]` for numeric computation with small values
- Use `rw [function_name]` to rewrite in hypotheses

## Proof Structure for Recursive Functions

For a recursively-defined function, proving three key properties in this order avoids circular dependencies:
1. **Bound lemma** (`imo_f_bound`): Constrains output size relative to input
2. **Functional equation** (`imo_f_functional`): Uses bound lemma to establish inductive hypotheses
3. **Monotonicity** (`imo_f_lt_succ`): Uses bound to compare consecutive values

## Zeckendorf Representation is Proof-Friendly

The Zeckendorf decomposition (representing each ℕ as a sum of non-consecutive Fibonacci numbers) is:
- Well-founded for recursion (guarantees termination via decreasing Zeckendorf indices)
- Computationally compact (avoids large intermediate values)
- Mathematically elegant (Fibonacci identities fib(k+2) = fib(k+1) + fib(k) directly establish the functional equation)

The proof that `f(f(n)) = f(n) + n` follows naturally from Zeckendorf properties and Fibonacci recurrence.
