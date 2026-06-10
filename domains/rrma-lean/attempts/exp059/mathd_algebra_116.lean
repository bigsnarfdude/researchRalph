import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_116 (k x : ℝ) (h₀ : x = (13 - Real.sqrt 131) / 4)
    (h₁ : 2 * x ^ 2 - 13 * x + k = 0) : k = 19 / 4 := by
  have hsq : Real.sqrt 131 ^ 2 = 131 := Real.sq_sqrt (by norm_num : (131:ℝ) ≥ 0)
  nlinarith [h₀, h₁, hsq]
