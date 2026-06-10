import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

-- x + 2/x = y + 2/y → (x-y) + 2(1/x - 1/y) = 0 → (x-y) + 2(y-x)/(xy) = 0
-- → (x-y)(1 - 2/(xy)) = 0. Since x≠y, xy = 2.
theorem amc12a_2013_p8 (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : x ≠ y)
  (h₃ : x + 2 / x = y + 2 / y) : x * y = 2 := by
  have hxy : x * y ≠ 0 := mul_ne_zero h₀ h₁
  have key : (x - y) * (1 - 2 / (x * y)) = 0 := by field_simp at h₃ ⊢; nlinarith
  have : x - y ≠ 0 := sub_ne_zero.mpr h₂
  have : 1 - 2 / (x * y) = 0 := by
    rcases mul_eq_zero.mp key with h | h
    · exact absurd h this
    · exact h
  field_simp at this
  linarith
