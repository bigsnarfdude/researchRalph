import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_206 (a b : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = x ^ 2 + a * x + b) (h₁ : 2 * a ≠ b)
  (h₂ : f (2 * a) = 0) (h₃ : f b = 0) : a + b = -1 := by
  simp only [h₀] at h₂ h₃
  -- h₂: 4a²+2a²+b=0 → b=-6a²  h₃: b²+ab+b=0 → b(b+a+1)=0
  have hb3 : b * (b + a + 1) = 0 := by nlinarith
  rcases mul_eq_zero.mp hb3 with hb0 | hab
  · -- b=0, then 6a²=0 → a=0, but 2*0=0=b contradicts h₁
    exfalso; apply h₁; nlinarith
  · linarith
