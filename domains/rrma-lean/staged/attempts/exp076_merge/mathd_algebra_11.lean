import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_11 (a b : ℝ) (h₀ : a ≠ b) (h₁ : a ≠ 2 * b)
    (h₂ : (4 * a + 3 * b) / (a - 2 * b) = 5) : (a + 11 * b) / (a - b) = 2 := by
  have hab : a - 2 * b ≠ 0 := sub_ne_zero.mpr h₁
  rw [div_eq_iff hab] at h₂
  have ha : a = 13 * b := by linarith
  have hb : b ≠ 0 := by intro hb; rw [hb, mul_zero] at ha; rw [ha, hb] at h₀; exact h₀ rfl
  have hab2 : a - b ≠ 0 := by
    rw [ha]; intro h
    have : 12 * b = 0 := by linarith
    have : b = 0 := by linarith [mul_eq_zero.mp this]
    exact hb this
  rw [div_eq_iff hab2, ha]; ring
