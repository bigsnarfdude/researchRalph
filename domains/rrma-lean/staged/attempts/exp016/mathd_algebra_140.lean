import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_140 (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : ∀ x, 24 * x ^ 2 - 19 * x - 35 = (a * x - 5) * (2 * (b * x) + c)) : a * b - 3 * c = -9 := by
  have h2 := h₁ 0
  have h3 := h₁ 1
  have h4 := h₁ (-1)
  have h5 := h₁ 2
  simp only [mul_zero, zero_add, sub_zero, mul_one, one_mul, mul_neg, neg_mul] at *
  nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg c, sq_nonneg (a - b), h₀.1, h₀.2.1, h₀.2.2,
             mul_pos h₀.1 h₀.2.1, mul_pos h₀.1 h₀.2.2, mul_pos h₀.2.1 h₀.2.2]
