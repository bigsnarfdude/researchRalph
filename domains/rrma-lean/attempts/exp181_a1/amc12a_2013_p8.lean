import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem amc12a_2013_p8 (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : x ≠ y)
  (h₃ : x + 2 / x = y + 2 / y) : x * y = 2 := by
  have : (x - y) * (1 - 2 / (x * y)) = 0 := by field_simp at h₃ ⊢; linarith
  have : 1 - 2 / (x * y) = 0 := by
    rcases mul_eq_zero.mp this with h | h
    · exact absurd h (sub_ne_zero.mpr h₂)
    · exact h
  field_simp at this
  linarith
