import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_510 (x y : ℝ) (h₀ : x + y = 13) (h₁ : x * y = 24) :
  Real.sqrt (x ^ 2 + y ^ 2) = 11 := by
  have hsq : x ^ 2 + y ^ 2 = 121 := by nlinarith
  rw [hsq, show (121:ℝ) = 11^2 from by norm_num, Real.sqrt_sq (by norm_num : (11:ℝ) ≥ 0)]
