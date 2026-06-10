import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

-- 8(a⁴+b⁴) - (a+b)⁴ = (a-b)⁴ + 6(a-b)²(a+b)² ≥ 0
theorem algebra_apb4leq8ta4pb4 (a b : ℝ) (h₀ : 0 < a ∧ 0 < b) : (a + b) ^ 4 ≤ 8 * (a ^ 4 + b ^ 4) := by
  nlinarith [sq_nonneg ((a - b) ^ 2), sq_nonneg ((a - b) * (a + b)), sq_nonneg (a - b),
             sq_nonneg (a * b), h₀.1, h₀.2]
