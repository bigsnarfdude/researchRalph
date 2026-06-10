import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem amc12b_2003_p17 (x y : ℝ) (h₀ : 0 < x ∧ 0 < y) (h₁ : Real.log (x * y ^ 3) = 1)
  (h₂ : Real.log (x ^ 2 * y) = 1) : Real.log (x * y) = 3 / 5 := by
  have hx := h₀.1; have hy := h₀.2
  rw [Real.log_mul (ne_of_gt hx) (ne_of_gt (pow_pos hy 3))] at h₁
  rw [Real.log_pow] at h₁
  rw [Real.log_mul (ne_of_gt (pow_pos hx 2)) (ne_of_gt hy)] at h₂
  rw [Real.log_pow] at h₂
  rw [Real.log_mul (ne_of_gt hx) (ne_of_gt hy)]
  push_cast at h₁ h₂ ⊢; linarith
