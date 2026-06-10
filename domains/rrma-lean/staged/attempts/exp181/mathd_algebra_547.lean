import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_547 (x y : ℝ) (h₀ : x = 5) (h₁ : y = 2) : Real.sqrt (x ^ 3 - 2 ^ y) = 11 := by
  rw [h₀, h₁]; norm_num
