import Mathlib

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Rat

theorem amc12a_2009_p9 (a b c : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f (x + 3) = 3 * x ^ 2 + 7 * x + 4)
  (h₁ : ∀ x, f x = a * x ^ 2 + b * x + c) : a + b + c = 2 := by
  have h2 : f 1 = a + b + c := by rw [h₁]; ring
  have h3 : f 1 = 2 := by
    have := h₀ (-2)
    norm_num at this
    exact this
  linarith
