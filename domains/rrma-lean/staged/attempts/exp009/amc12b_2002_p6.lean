import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

-- x² + ax + b = (x-a)(x-b) = x² - (a+b)x + ab for all x
-- Comparing coefficients: a = -(a+b), b = ab
-- From b = ab: b(a-1)=0, b≠0 so a=1. Then 1 = -(1+b) → b=-2.
theorem amc12b_2002_p6 (a b : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0)
  (h₁ : ∀ x, x ^ 2 + a * x + b = (x - a) * (x - b)) : a = 1 ∧ b = -2 := by
  have h_eq : ∀ x : ℝ, a * x + b = -(a + b) * x + a * b := by
    intro x
    have := h₁ x
    nlinarith [this, sq_nonneg x]
  -- Evaluate at x=0 and x=1
  have h0 := h_eq 0
  simp at h0
  have h1 := h_eq 1
  simp at h1
  have hb := h₀.2
  constructor
  · -- b = ab → b(a-1) = 0 → a = 1
    have : b * (a - 1) = 0 := by nlinarith
    rcases mul_eq_zero.mp this with h | h
    · exact absurd h hb
    · linarith
  · -- a = 1 → a = -(1+b) → b = -2
    have ha : a = 1 := by
      have : b * (a - 1) = 0 := by nlinarith
      rcases mul_eq_zero.mp this with h | h
      · exact absurd h hb
      · linarith
    nlinarith [ha]
