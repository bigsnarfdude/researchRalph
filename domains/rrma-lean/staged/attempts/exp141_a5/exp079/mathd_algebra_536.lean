import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_536 : ↑3! * ((2 : ℝ) ^ 3 + Real.sqrt 9) / 2 = (33 : ℝ) := by
  rw [show (9:ℝ) = 3^2 from by norm_num, Real.sqrt_sq (by norm_num : (3:ℝ) ≥ 0)]
  norm_num
