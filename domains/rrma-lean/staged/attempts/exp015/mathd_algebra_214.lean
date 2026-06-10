import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_214 (a : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = a * (x - 2) ^ 2 + 3) (h₁ : f 4 = 4) :
  f 6 = 7 := by
  simp only [h₀] at *; linarith