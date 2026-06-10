import Mathlib

set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

theorem aimeI_2000_p7 (x y z : ℝ) (m : ℚ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x * y * z = 1)
  (h₂ : x + 1 / z = 5) (h₃ : y + 1 / x = 29) (h₄ : z + 1 / y = m) (h₅ : 0 < m) :
  ↑m.den + m.num = 5 := by
  have hxp := h₀.1; have hyp := h₀.2.1; have hzp := h₀.2.2
  -- 1/z = xy (from xyz=1)
  have hz_inv : 1 / z = x * y := by
    have hzne : z ≠ 0 := ne_of_gt hzp
    rw [div_eq_iff hzne]; linarith
  -- x + xy = 5
  have hxy_sum : x * (1 + y) = 5 := by nlinarith
  -- xy + 1 = 29x (multiply y + 1/x = 29 by x)
  have hxy_prod : x * y + 1 = 29 * x := by
    have hxne : x ≠ 0 := ne_of_gt hxp
    have h := h₃
    have : x * (y + 1 / x) = x * 29 := by rw [h]
    rw [mul_add, mul_div_cancel₀ 1 hxne] at this
    linarith
  -- x = 1/5 (from 5-x+1=29x)
  have hx : x = 1 / 5 := by
    have : x + x * y = 5 := by nlinarith [hxy_sum]
    have : x * y = 5 - x := by linarith
    linarith
  have hy : y = 24 := by nlinarith [hxy_prod]
  have hz : z = 5 / 24 := by
    have hzne : z ≠ 0 := ne_of_gt hzp
    rw [hx, hy] at h₁; field_simp at h₁ ⊢; linarith
  have hm_val : (m : ℝ) = 1 / 4 := by
    rw [hz, hy] at h₄
    have hyne : (24 : ℝ) ≠ 0 := by norm_num
    field_simp at h₄; linarith
  have : m = 1 / 4 := by
    have : ((m : ℚ) : ℝ) = ((1 / 4 : ℚ) : ℝ) := by push_cast; linarith
    exact_mod_cast this
  rw [this]; norm_num
