# Mistakes — agent2

## Attempted omega directly on digit constraints (2026-05-26, exp006-exp008)
- **What:** Tried `omega` directly on goals involving `Nat.digits` constraints
- **Result:** omega could not prove the goal; reported "counterexample may satisfy constraints 0 ≤ c ≤ 62"
- **Lesson:** omega is an arithmetic solver for linear constraints only. It cannot reason about function calls like `Nat.digits` directly. Must first convert digit constraints to arithmetic bounds using finite computation.

## Tried smaller witnesses (2026-05-26, exp002)
- **What:** Attempted witness n=7 instead of n=62, hoping omega would find it easier
- **Result:** Same error; omega still couldn't handle digit constraints
- **Lesson:** The witness choice doesn't matter if the fundamental approach is wrong. Fix the tactic, not the witness.
