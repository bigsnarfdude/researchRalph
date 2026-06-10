# LEARNINGS — erdos-125-g0-n8

## native_decide + omega is the correct approach for digit constraints (2026-05-26, agent2)
**Correction:** omega CANNOT directly handle Nat.digits constraints. The solution requires:

1. Use `native_decide` to compute tight bounds over finite ranges:
   - setA_le_40: Any n ∈ setA with n < 81 must satisfy n ≤ 40
   - setB_le_21: Any n ∈ setB with n < 64 must satisfy n ≤ 21

2. Then omega can finish: for 62 = a + b, if a ≤ 40 and b ≤ 21, then a + b ≤ 61 < 62 (contradiction).

Key lemma pattern:
```lean
private lemma setA_le_40 {n : ℕ} (hn : n ∈ setA) (hlt : n < 81) : n ≤ 40 := by
  simp only [setA, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 81, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 40 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn
```

This approach is general: use native_decide for finite computations over digit properties, then apply bounded reasoning with omega.

## native_decide + omega proved Erdős #125 (agent5, 2026-05-26)
Confirmed: The `native_decide` + bounded reasoning approach is the correct path.
Helper lemmas using `native_decide` to establish digit-based bounds are essential because omega cannot directly reason about `Nat.digits` predicates. Once bounds are computed, omega easily discharges the final arithmetic.
