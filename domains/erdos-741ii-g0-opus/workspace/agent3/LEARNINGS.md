# LEARNINGS — agent3

## Math
- A basis of order 2 ⇒ A+A ⊇ [4,∞) ⇒ A+A syndetic. So part 2 (one monochromatic
  sumset non-syndetic) is never automatic; it is the whole content.
- A subset of a non-syndetic set is non-syndetic (bigger gaps). So if A+A were
  non-syndetic both parts would be too — but basis forbids that.
- The adversary's universal weapon: take any arithmetically rich colour class and
  refine it by a finer modulus so BOTH classes have syndetic self-sumsets
  (the even/odd attack on ℕ, the mod-4 attack on evens, the within-block parity
  attack on interval blocks). Defeating it needs lacunarity at every scale, which
  conflicts with the basis density requirement (~√n).

## Lean / Mathlib
- `Nat.pow_log_le_self 4 (hn0 : n ≠ 0) : 4 ^ Nat.log 4 n ≤ n`.
- `Nat.lt_pow_succ_log_self (by norm_num : 1 < 4) n : n < 4 ^ (Nat.log 4 n + 1)`.
- `rw [pow_succ] at hlt` turns `4^(k+1)` into `4^k * 4`; then `omega` treats
  `4^k` as an opaque atom and solves all block-membership inequalities.
- `le_or_lt` failed under `rcases` here ("not an inductive datatype") — use
  `by_cases` instead. (Matches the known le_or_lt gotcha for this Mathlib.)
- Set-builder membership goals: discharge with
  `by simp only [Set.mem_setOf_eq]; omega`.
- `Even 0` = `⟨0, rfl⟩`; `Even 2` = `⟨1, rfl⟩`.
