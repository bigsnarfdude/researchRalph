import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_131 (a b : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = 2 * x ^ 2 - 7 * x + 2)
  (h₁ : f a = 0) (h₂ : f b = 0) (h₃ : a ≠ b) : 1 / (a - 1) + 1 / (b - 1) = -1 := by
  simp only [h₀] at h₁ h₂
  have hsum : a + b = 7 / 2 := by
    have : (a - b) * (2 * (a + b) - 7) = 0 := by nlinarith
    rcases mul_eq_zero.mp this with h | h
    · exact absurd (sub_eq_zero.mp h) h₃
    · linarith
  have hab : a * b = 1 := by nlinarith
  have ha1 : a ≠ 1 := by intro h; subst h; linarith
  have hb1 : b ≠ 1 := by intro h; subst h; linarith
  have ha1' : a - 1 ≠ 0 := sub_ne_zero.mpr ha1
  have hb1' : b - 1 ≠ 0 := sub_ne_zero.mpr hb1
  rw [div_add_div _ _ ha1' hb1', div_eq_iff (mul_ne_zero ha1' hb1')]
  nlinarith
