import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_11 (a b : ℝ) (h₀ : a ≠ b) (h₁ : a ≠ 2 * b)
    (h₂ : (4 * a + 3 * b) / (a - 2 * b) = 5) : (a + 11 * b) / (a - b) = 2 := by
  have hab : a - 2 * b ≠ 0 := sub_ne_zero.mpr h₁
  have h3 : 4 * a + 3 * b = 5 * (a - 2 * b) := by
    field_simp at h₂
    linarith
  have h4 : a = 13 * b := by linarith
  have hab2 : a - b ≠ 0 := by
    rw [h4]; ring_nf; intro h; apply h₀; linarith
  field_simp
  linarith
