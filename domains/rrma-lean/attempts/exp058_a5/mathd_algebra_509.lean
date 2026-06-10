import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_509 :
  Real.sqrt ((5 / Real.sqrt 80 + Real.sqrt 845 / 9 + Real.sqrt 45) / Real.sqrt 5) = 13 / 6 := by
  have h5 : Real.sqrt 5 > 0 := Real.sqrt_pos.mpr (by norm_num)
  have h5_sq : Real.sqrt 5 ^ 2 = 5 := Real.sq_sqrt (by norm_num)
  -- sqrt(80) = 4*sqrt(5)
  have h80 : Real.sqrt 80 = 4 * Real.sqrt 5 := by
    rw [show (80:ℝ) = 4^2 * 5 from by norm_num, Real.sqrt_mul (by norm_num : (0:ℝ) ≤ 4^2),
        Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 4)]
  -- sqrt(845) = 13*sqrt(5)
  have h845 : Real.sqrt 845 = 13 * Real.sqrt 5 := by
    rw [show (845:ℝ) = 13^2 * 5 from by norm_num, Real.sqrt_mul (by norm_num : (0:ℝ) ≤ 13^2),
        Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 13)]
  -- sqrt(45) = 3*sqrt(5)
  have h45 : Real.sqrt 45 = 3 * Real.sqrt 5 := by
    rw [show (45:ℝ) = 3^2 * 5 from by norm_num, Real.sqrt_mul (by norm_num : (0:ℝ) ≤ 3^2),
        Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 3)]
  rw [h80, h845, h45]
  -- Simplify the expression inside sqrt
  have h_inner : (5 / (4 * Real.sqrt 5) + 13 * Real.sqrt 5 / 9 + 3 * Real.sqrt 5) / Real.sqrt 5 = 169 / 36 := by
    have h5_ne : Real.sqrt 5 ≠ 0 := ne_of_gt h5
    field_simp
    nlinarith [h5_sq]
  rw [h_inner]
  -- sqrt(169/36) = 13/6
  rw [show (169:ℝ)/36 = (13/6)^2 from by ring, Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 13/6)]
