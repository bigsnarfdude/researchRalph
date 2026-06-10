import Mathlib

set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

theorem algebra_amgm_faxinrrp2msqrt2geq2mxm1div2x :
  ∀ x > 0, 2 - Real.sqrt 2 ≥ 2 - x - 1 / (2 * x) := by
  intro x hx
  have hx_pos : (0 : ℝ) < x := hx
  have h2x_pos : (0 : ℝ) < 2 * x := by linarith
  have hsqrt_x := Real.sq_sqrt (le_of_lt hx_pos)
  have hsqrt_2 := Real.sq_sqrt (show (0:ℝ) ≤ 2 by norm_num)
  have hsqrt_x_pos : Real.sqrt x > 0 := Real.sqrt_pos.mpr hx_pos
  have h_inv_pos : 1 / (2 * x) > 0 := by positivity
  have hsqrt_inv := Real.sq_sqrt (le_of_lt h_inv_pos)
  have hsqrt_inv_pos : Real.sqrt (1 / (2 * x)) > 0 := Real.sqrt_pos.mpr h_inv_pos
  -- Key: √x · √(1/(2x)) = √(1/2)
  have h_prod : Real.sqrt x * Real.sqrt (1 / (2 * x)) = Real.sqrt (1/2) := by
    rw [← Real.sqrt_mul (le_of_lt hx_pos)]
    congr 1; field_simp
  -- √(1/2) · 2 = √2 (so 2·√x·√(1/(2x)) = √2)
  have h_sqrt_half : 2 * Real.sqrt (1/2) = Real.sqrt 2 := by
    rw [show (1:ℝ)/2 = 2/4 from by ring, show (2:ℝ)/4 = 2/2^2 from by ring,
        Real.sqrt_div' 2 (by norm_num : (0:ℝ) ≤ 2^2), Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 2)]
    ring
  nlinarith [sq_nonneg (Real.sqrt x - Real.sqrt (1 / (2 * x)))]
