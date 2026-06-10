import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem amc12_2001_p9 (f : ℝ → ℝ) (h₀ : ∀ x > 0, ∀ y > 0, f (x * y) = f x / y) (h₁ : f 500 = 3) :
    f 600 = 5 / 2 := by
  have h := h₀ 500 (by norm_num) (6/5) (by norm_num)
  simp only [show (500:ℝ) * (6/5) = 600 from by ring] at h
  rw [h₁] at h; linarith
