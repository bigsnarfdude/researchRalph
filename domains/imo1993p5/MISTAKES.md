# Mistakes — imo1993p5

## Pattern Matching + rfl Fails

**What**: Tried to use `rfl` in:
```lean
@[simp] private lemma imo_f_zero : imo_f 0 = 0 := rfl
```

**Result**: Type error — `rfl` cannot prove definitional equality when the function is defined via pattern matching because Lean doesn't automatically see pattern constructors as definitionally equal.

**Lesson**: Pattern-matched definitions need `simp [imo_f]` or `unfold imo_f` to establish equality, not `rfl` alone.

---

## Over-Aggressive simp Solves Goals

**What**: After `simp`, attempted to apply an `exact` tactic:
```lean
rcases Nat.eq_zero_or_pos n with rfl | hpos
· simp; exact fib_pos.mpr (Nat.succ_pos k)
```

**Result**: "No goals to be solved" — the `simp` completely discharged the goal, leaving nothing for `exact`.

**Lesson**: Use `simp only [specific_lemmas]` to limit simp's range, or remove the redundant tactic. When in doubt, use `simp?` to see what simp is actually doing.

---

## Incorrect Type Application in Base Case

**What**: Tried to prove `0 < fib(greatestFib 1 + 1)` by applying `fib_pos.mpr (greatestFib_pos.mpr ...)` directly.

**Result**: Type mismatch — the lemmas don't directly combine.

**Lesson**: For concrete base cases, use `norm_num` with the relevant lemmas to compute numerically: `norm_num [fib, greatestFib]`.
