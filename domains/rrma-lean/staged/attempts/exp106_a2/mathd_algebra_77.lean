import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_77 (a b : ℝ) (f : ℝ → ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : a ≠ b)
  (h₂ : ∀ x, f x = x ^ 2 + a * x + b) (h₃ : f a = 0) (h₄ : f b = 0) : a = 1 ∧ b = -2 := by
  simp only [h₂] at h₃ h₄
  -- h₃: a²+a²+b=0 → 2a²+b=0. h₄: b²+ab+b=0 → b(b+a+1)=0
  have hb : b * (b + a + 1) = 0 := by nlinarith
  rcases mul_eq_zero.mp hb with hb0 | hab
  · exact absurd hb0 h₀.2
  · -- b+a+1=0 → b = -a-1. Substitute into 2a²+b=0: 2a²-a-1=0 → (2a+1)(a-1)=0
    have hba : b = -a - 1 := by linarith
    have : 2 * a ^ 2 - a - 1 = 0 := by nlinarith
    have : (2 * a + 1) * (a - 1) = 0 := by nlinarith
    rcases mul_eq_zero.mp this with h | h
    · -- a=-1/2, b=-1/2, but a≠b! Actually a=-1/2, b=-(-1/2)-1=-1/2. Contradiction.
      exfalso; apply h₁; linarith [hba]
    · constructor; linarith; linarith [hba]
