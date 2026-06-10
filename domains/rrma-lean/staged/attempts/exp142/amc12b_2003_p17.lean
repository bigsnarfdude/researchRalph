import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem amc12b_2003_p17 (x y : ℝ) (h₀ : 0 < x ∧ 0 < y) (h₁ : Real.log (x * y ^ 3) = 1)
  (h₂ : Real.log (x ^ 2 * y) = 1) : Real.log (x * y) = 3 / 5 := by
  have h1 : Real.log x + 3 * Real.log y = 1 := by
    rw [← h₁, Real.log_mul h₀.1.ne' (pow_pos h₀.2 3).ne', Real.log_pow]
    push_cast; ring
  have h2 : 2 * Real.log x + Real.log y = 1 := by
    rw [← h₂, Real.log_mul (pow_pos h₀.1 2).ne' h₀.2.ne', Real.log_pow]
    push_cast; ring
  rw [Real.log_mul h₀.1.ne' h₀.2.ne']
  linarith
