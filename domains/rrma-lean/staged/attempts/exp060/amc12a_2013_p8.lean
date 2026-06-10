import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem amc12a_2013_p8 (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : x ≠ y)
  (h₃ : x + 2 / x = y + 2 / y) : x * y = 2 := by
  field_simp at h₃
  have : (x - y) * (x * y - 2) = 0 := by nlinarith
  rcases mul_eq_zero.mp this with h | h
  · exact absurd (sub_eq_zero.mp h) h₂
  · linarith
