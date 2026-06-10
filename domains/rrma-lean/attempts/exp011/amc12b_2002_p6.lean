import Mathlib

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Rat

theorem amc12b_2002_p6 (a b : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0)
  (h₁ : ∀ x, x ^ 2 + a * x + b = (x - a) * (x - b)) : a = 1 ∧ b = -2 := by
  have h2 := h₁ 0  -- b = a * b
  have h3 := h₁ 1  -- 1 + a + b = (1-a)(1-b)
  simp at h2  -- b = a * b
  have ha : a = 1 := by
    have : b * (1 - a) = 0 := by nlinarith
    cases mul_eq_zero.mp this with
    | inl hb => exact absurd hb h₀.2
    | inr ha1 => linarith
  constructor
  · exact ha
  · nlinarith
