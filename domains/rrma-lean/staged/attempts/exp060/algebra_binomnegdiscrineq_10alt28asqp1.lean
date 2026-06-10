import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem algebra_binomnegdiscrineq_10alt28asqp1 (a : ℝ) : 10 * a ≤ 28 * a ^ 2 + 1 := by
  nlinarith [sq_nonneg (28*a - 5)]
