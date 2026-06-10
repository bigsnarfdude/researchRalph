# Desires — imo1993p5

## Interactive Lean Environment with Hover Tactic Suggestions

Would have been helpful to have IDE support showing:
- What `simp` is actually doing (goal before/after)
- Which lemmas are triggering
- Goal state at each line

Lean 4 development is much slower with shell-based edit/compile/check loops.

## Mathlib Lemma Discovery Tool

The proof relies heavily on Fibonacci properties (fib_pos, fib_add_two, fib_mono, etc.). Searching Mathlib source by hand is tedious. A tool that finds lemmas by goal type would accelerate proof search.

## Automatic Pattern Matching Unfolding

Lean 4 doesn't automatically treat pattern-matched definitions as unfolded in `rfl` context. A tactic that handles this automatically (beyond just `simp`) would reduce trial-and-error.
