import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem amc12b_2002_p6 (a b : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0)
  (h₁ : ∀ x, x ^ 2 + a * x + b = (x - a) * (x - b)) : a = 1 ∧ b = -2 := by
  have h0 := h₁ 0; simp at h0  -- b = a*b
  have h1 := h₁ 1; ring_nf at h1  -- 1+a+b = (1-a)(1-b) = 1-a-b+ab
  have ha : a = 1 := by
    have : b * (a - 1) = 0 := by linarith [h0]
    rcases mul_eq_zero.mp this with h | h
    · exact absurd h h₀.2
    · linarith
  constructor
  · exact ha
  · nlinarith [h0]
