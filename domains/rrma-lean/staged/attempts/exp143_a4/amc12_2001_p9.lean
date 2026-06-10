import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem amc12_2001_p9 (f : ℝ → ℝ) (h₀ : ∀ x > 0, ∀ y > 0, f (x * y) = f x / y) (h₁ : f 500 = 3) :
    f 600 = 5 / 2 := by
  have h2 : f 600 = f 500 / (6/5) := by
    have := h₀ 500 (by positivity) (6/5) (by positivity)
    simp only [show (500 : ℝ) * (6 / 5) = 600 from by ring] at this
    exact this
  rw [h2, h₁]
  ring
