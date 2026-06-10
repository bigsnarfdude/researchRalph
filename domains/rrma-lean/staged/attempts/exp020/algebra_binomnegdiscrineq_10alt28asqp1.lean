import Mathlib

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Rat

-- 28a² - 10a + 1 ≥ 0: discriminant = 100-112 < 0, always true.
-- Witness: (28a-5)² = 784a²-280a+25 ≥ 0, so 28(28a²-10a+1) = (28a-5)²+3 ≥ 3 > 0
theorem algebra_binomnegdiscrineq_10alt28asqp1 (a : ℝ) : 10 * a ≤ 28 * a ^ 2 + 1 := by
  nlinarith [sq_nonneg (28 * a - 5)]
