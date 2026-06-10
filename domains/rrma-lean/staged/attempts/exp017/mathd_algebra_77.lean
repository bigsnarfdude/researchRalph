import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_77 (a b : ℝ) (f : ℝ → ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : a ≠ b)
  (h₂ : ∀ x, f x = x ^ 2 + a * x + b) (h₃ : f a = 0) (h₄ : f b = 0) : a = 1 ∧ b = -2 := by
  simp only [h₂] at h₃ h₄
  -- h₃: a² + a² + b = 0 → 2a² + b = 0
  -- h₄: b² + ab + b = 0 → b(b + a + 1) = 0
  -- b ≠ 0 so b + a + 1 = 0 → b = -a - 1
  -- 2a² + (-a-1) = 0 → 2a² - a - 1 = 0 → (2a+1)(a-1) = 0
  -- a ≠ 0 and checking: a = 1 → b = -2, a = -1/2 → b = -1/2 but a ≠ b so a = -1/2 works
  -- but then a = -1/2, b = -1/2 contradicts a ≠ b
  have hb : b * (b + a + 1) = 0 := by nlinarith
  have hba : b + a + 1 = 0 := by
    rcases mul_eq_zero.mp hb with h | h
    · exact absurd h h₀.2
    · exact h
  have hab : 2 * a ^ 2 - a - 1 = 0 := by nlinarith
  have : (2 * a + 1) * (a - 1) = 0 := by nlinarith
  rcases mul_eq_zero.mp this with h | h
  · exfalso; apply h₁
    have : a = -1/2 := by linarith
    have : b = -1/2 := by linarith
    linarith
  · constructor
    · linarith
    · linarith
