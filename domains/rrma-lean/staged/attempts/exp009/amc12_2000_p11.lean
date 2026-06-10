import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12_2000_p11 (a b : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : a * b = a - b) :
    a / b + b / a - a * b = 2 := by
  have ha : a ≠ 0 := h₀.1
  have hb : b ≠ 0 := h₀.2
  field_simp
  nlinarith [h₁, sq_nonneg a, sq_nonneg b, sq_nonneg (a - b)]
