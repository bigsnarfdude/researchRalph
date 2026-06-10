import Mathlib

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Rat

theorem amc12a_2013_p8 (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : x ≠ y)
  (h₃ : x + 2 / x = y + 2 / y) : x * y = 2 := by
  have hx := h₀
  have hy := h₁
  have hxy := h₂
  -- x + 2/x = y + 2/y → (x-y) + 2(1/x - 1/y) = 0
  -- → (x-y) - 2(x-y)/(xy) = 0 → (x-y)(1 - 2/(xy)) = 0
  -- Since x ≠ y, xy = 2
  have h4 : x - y ≠ 0 := sub_ne_zero.mpr h₂
  field_simp at h₃
  -- After field_simp, we should have a polynomial equation
  have h5 : (x - y) * (x * y - 2) = 0 := by nlinarith
  rcases mul_eq_zero.mp h5 with h6 | h6
  · exact absurd h6 h4
  · linarith
