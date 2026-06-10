import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_131 (a b : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = 2 * x ^ 2 - 7 * x + 2)
  (h₁ : f a = 0) (h₂ : f b = 0) (h₃ : a ≠ b) : 1 / (a - 1) + 1 / (b - 1) = -1 := by
  simp only [h₀] at h₁ h₂
  -- h₁ : 2 * a ^ 2 - 7 * a + 2 = 0
  -- h₂ : 2 * b ^ 2 - 7 * b + 2 = 0
  -- f(1) = 2 - 7 + 2 = -3 ≠ 0, so a ≠ 1 and b ≠ 1
  have ha1 : a ≠ 1 := by intro h; rw [h] at h₁; norm_num at h₁
  have hb1 : b ≠ 1 := by intro h; rw [h] at h₂; norm_num at h₂
  have hane : a - 1 ≠ 0 := sub_ne_zero.mpr ha1
  have hbne : b - 1 ≠ 0 := sub_ne_zero.mpr hb1
  -- a and b are roots of 2x²-7x+2=0. Subtracting: 2(a²-b²)-7(a-b)=0 → (a-b)(2(a+b)-7)=0
  -- Since a≠b, 2(a+b)=7 → a+b=7/2
  -- From 2a²-7a+2=0: 2a(a)+2=7a → product ab: multiply eqs gives 4a²b²-7*2ab*(a+b)/...
  -- Actually: from 2a²=7a-2 and 2b²=7b-2, we get a+b=7/2 and ab=1 by symmetric functions
  have hab_diff : (a - b) * (2 * (a + b) - 7) = 0 := by ring_nf; nlinarith
  have hne : a - b ≠ 0 := sub_ne_zero.mpr h₃
  have hab_sum : a + b = 7 / 2 := by
    have := mul_eq_zero.mp hab_diff
    cases this with
    | inl h => exact absurd h hne
    | inr h => linarith
  have hab_prod : a * b = 1 := by nlinarith
  field_simp
  nlinarith [mul_comm a b]