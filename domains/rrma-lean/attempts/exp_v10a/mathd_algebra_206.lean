import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_206 (a b : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = x ^ 2 + a * x + b) (h₁ : 2 * a ≠ b)
  (h₂ : f (2 * a) = 0) (h₃ : f b = 0) : a + b = -1 := by
  simp only [h₀] at h₂ h₃
  have hb : b * (b + a + 1) = 0 := by ring_nf; linarith
  rcases mul_eq_zero.mp hb with hb0 | hba
  · exfalso; apply h₁
    have ha : a = 0 := by nlinarith [sq_nonneg a]
    rw [ha, hb0]; ring
  · linarith
