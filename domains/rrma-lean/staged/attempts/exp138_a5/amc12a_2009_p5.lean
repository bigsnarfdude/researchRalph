import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem amc12a_2009_p5 (x : ℝ) (h₀ : x ^ 3 - (x + 1) * (x - 1) * x = 5) : x ^ 3 = 125 := by
  have : x = 5 := by nlinarith [sq_nonneg x]
  subst this; norm_num
