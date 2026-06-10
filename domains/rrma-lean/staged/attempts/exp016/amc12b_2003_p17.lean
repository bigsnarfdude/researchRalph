import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12b_2003_p17 (x y : ℝ) (h₀ : 0 < x ∧ 0 < y) (h₁ : Real.log (x * y ^ 3) = 1)
  (h₂ : Real.log (x ^ 2 * y) = 1) : Real.log (x * y) = 3 / 5 := by
  have hx : 0 < x := h₀.1
  have hy : 0 < y := h₀.2
  rw [Real.log_mul (ne_of_gt hx) (ne_of_gt (pow_pos hy 3)), Real.log_pow] at h₁
  rw [Real.log_mul (ne_of_gt (pow_pos hx 2)) (ne_of_gt hy), Real.log_pow] at h₂
  rw [Real.log_mul (ne_of_gt hx) (ne_of_gt hy)]
  -- h₁: log x + 3 * log y = 1, h₂: 2 * log x + log y = 1
  -- h₁ + 2*h₂ = 5*(log x + log y) = 3
  linear_combination (1 / 5) * h₁ + (2 / 5) * h₂
