import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_267 (x : ℝ) (h₀ : x ≠ 1) (h₁ : x ≠ -2)
  (h₂ : (x + 1) / (x - 1) = (x - 2) / (x + 2)) : x = 0 := by
  have h3 : x - 1 ≠ 0 := sub_ne_zero.mpr h₀
  have h4 : x + 2 ≠ 0 := by intro h; exact h₁ (by linarith)
  have h5 : (x + 1) * (x + 2) = (x - 2) * (x - 1) := by
    rw [div_eq_div_iff h3 h4] at h₂; linarith
  nlinarith
