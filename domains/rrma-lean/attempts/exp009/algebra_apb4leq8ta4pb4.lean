import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_apb4leq8ta4pb4 (a b : ℝ) (h₀ : 0 < a ∧ 0 < b) : (a + b) ^ 4 ≤ 8 * (a ^ 4 + b ^ 4) := by
  nlinarith [sq_nonneg (a - b), sq_nonneg (a^2 - b^2), sq_nonneg (a^2 + b^2),
             sq_nonneg (a*b), sq_nonneg a, sq_nonneg b,
             mul_pos h₀.1 h₀.2, sq_abs (a - b), sq_abs (a + b)]
