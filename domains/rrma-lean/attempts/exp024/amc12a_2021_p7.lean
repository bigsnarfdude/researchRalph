import Mathlib

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Rat

theorem amc12a_2021_p7 (x y : ℝ) : 1 ≤ (x * y - 1) ^ 2 + (x + y) ^ 2 := by
  -- (xy - 1)² + (x+y)² = x²y² - 2xy + 1 + x² + 2xy + y² = x²y² + x² + y² + 1
  -- = x²(y² + 1) + (y² + 1) = (x² + 1)(y² + 1) ≥ 1
  nlinarith [sq_nonneg (x * y - 1), sq_nonneg (x + y), sq_nonneg x, sq_nonneg y]
