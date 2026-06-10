import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem algebra_amgm_faxinrrp2msqrt2geq2mxm1div2x :
  ∀ x > 0, 2 - Real.sqrt 2 ≥ 2 - x - 1 / (2 * x) := by
  intro x hx
  have hx2 : 0 < 2 * x := by linarith
  rw [ge_iff_le, sub_le_sub_iff_left]
  rw [div_add_eq_add_div, div_le_div_iff hx2 (by positivity)]
  nlinarith [sq_nonneg (x - 1 / Real.sqrt 2), Real.sq_sqrt (by linarith : (2:ℝ) ≥ 0), sq_nonneg (Real.sqrt 2 * x - 1)]
