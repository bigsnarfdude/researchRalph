import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

-- (a-b)² ≥ 0 → a²+b² ≥ 2ab → 2 ≥ 2ab → ab ≤ 1
theorem algebra_sqineq_2unitcircatblt1 (a b : ℝ) (h₀ : a ^ 2 + b ^ 2 = 2) : a * b ≤ 1 := by
  nlinarith [sq_nonneg (a - b)]
