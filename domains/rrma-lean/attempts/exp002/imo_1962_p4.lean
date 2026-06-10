import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem imo_1962_p4 (S : Set ℝ)
    (h₀ : S = { x : ℝ | Real.cos x ^ 2 + Real.cos (2 * x) ^ 2 + Real.cos (3 * x) ^ 2 = 1 }) :
    S =
      { x : ℝ |
        ∃ m : ℤ,
          x = π / 2 + m * π ∨
            x = π / 4 + m * π / 2 ∨ x = π / 6 + m * π / 6 ∨ x = 5 * π / 6 + m * π / 6 } := by
  first
    | omega
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg h₀]
    | simp_all [*]
    | decide