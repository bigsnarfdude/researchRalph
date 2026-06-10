import Mathlib

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Rat

theorem algebra_sqineq_36azm9asqle36zsq (z a : ℝ) : 36 * (a * z) - 9 * a ^ 2 ≤ 36 * z ^ 2 := by
  nlinarith [sq_nonneg (3 * a - 6 * z)]
