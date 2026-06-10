import Mathlib
set_option maxHeartbeats 8000000

open Real

theorem aime_1990_p2 :
  (52 + 6 * Real.sqrt 43) ^ ((3 : ℝ) / 2) - (52 - 6 * Real.sqrt 43) ^ ((3 : ℝ) / 2) = 828 := by
  have hsq43 : Real.sqrt 43 ^ 2 = 43 := Real.sq_sqrt (by norm_num : (43:ℝ) ≥ 0)
  have hsq43_pos : 0 < Real.sqrt 43 := Real.sqrt_pos.mpr (by norm_num : (43:ℝ) > 0)
  have h3lt : (3:ℝ) < Real.sqrt 43 := by
    have : Real.sqrt 9 = 3 := by rw [show (9:ℝ) = 3^2 from by norm_num]; exact Real.sqrt_sq (by norm_num : (3:ℝ) ≥ 0)
    rw [← this]; exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
  have hplus : 52 + 6 * Real.sqrt 43 = (3 + Real.sqrt 43) ^ 2 := by nlinarith [hsq43]
  have hminus : 52 - 6 * Real.sqrt 43 = (Real.sqrt 43 - 3) ^ 2 := by nlinarith [hsq43]
  have hplus_nn : (0:ℝ) ≤ 3 + Real.sqrt 43 := by linarith [hsq43_pos]
  have hminus_nn : (0:ℝ) ≤ Real.sqrt 43 - 3 := by linarith
  rw [hplus, hminus]
  have step1 : ((3 + Real.sqrt 43) ^ 2) ^ ((3:ℝ)/2) = (3 + Real.sqrt 43) ^ 3 := by
    rw [← rpow_natCast (3 + Real.sqrt 43) 2, ← rpow_mul hplus_nn]; norm_num
  have step2 : ((Real.sqrt 43 - 3) ^ 2) ^ ((3:ℝ)/2) = (Real.sqrt 43 - 3) ^ 3 := by
    rw [← rpow_natCast (Real.sqrt 43 - 3) 2, ← rpow_mul hminus_nn]; norm_num
  rw [step1, step2]
  have : (3 + Real.sqrt 43) ^ 3 - (Real.sqrt 43 - 3) ^ 3 = 54 + 18 * Real.sqrt 43 ^ 2 := by ring
  rw [this, hsq43]; norm_num
