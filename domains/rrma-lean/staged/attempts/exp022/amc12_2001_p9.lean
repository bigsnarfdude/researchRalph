import Mathlib

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Rat

theorem amc12_2001_p9 (f : ℝ → ℝ) (h₀ : ∀ x > 0, ∀ y > 0, f (x * y) = f x / y) (h₁ : f 500 = 3) :
    f 600 = 5 / 2 := by
  have h2 : f 600 = f 500 / (6 / 5) := by
    have : (500 : ℝ) > 0 := by norm_num
    have : (6 / 5 : ℝ) > 0 := by norm_num
    have h3 := h₀ 500 (by norm_num : (500:ℝ) > 0) (6/5) (by norm_num : (6/5:ℝ) > 0)
    convert h3 using 1
    norm_num
  rw [h2, h₁]
  norm_num
