import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_267 (x : ℝ) (h₀ : x ≠ 1) (h₁ : x ≠ -2)
  (h₂ : (x + 1) / (x - 1) = (x - 2) / (x + 2)) : x = 0 := by
  have hne1 : x - 1 ≠ 0 := sub_ne_zero.mpr h₀
  have hne2 : x + 2 ≠ 0 := by intro h; apply h₁; linarith
  field_simp at h₂
  nlinarith
