import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem amc12b_2002_p6 (a b : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0)
  (h₁ : ∀ x, x ^ 2 + a * x + b = (x - a) * (x - b)) : a = 1 ∧ b = -2 := by
  have ha := h₀.1; have hb := h₀.2
  have h1 : ∀ x : ℝ, x ^ 2 + a * x + b = x ^ 2 - (a + b) * x + a * b := by
    intro x; have := h₁ x; nlinarith
  have hcoeff_a : a = -(a + b) := by have := h1 1; have h0 := h1 0; nlinarith
  have hcoeff_b : b = a * b := by have := h1 0; nlinarith
  constructor
  · have : b * (1 - a) = 0 := by nlinarith
    rcases mul_eq_zero.mp this with hb0 | ha1
    · exact absurd hb0 hb
    · linarith
  · have : a = 1 := by
      have : b * (1 - a) = 0 := by nlinarith
      rcases mul_eq_zero.mp this with hb0 | ha1
      · exact absurd hb0 hb
      · linarith
    linarith [hcoeff_a]
