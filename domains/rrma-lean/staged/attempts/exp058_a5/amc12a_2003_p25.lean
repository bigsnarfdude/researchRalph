import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2003_p25 (a b : ℝ) (f : ℝ → ℝ) (h₀ : 0 < b)
  (h₁ : ∀ x, f x = Real.sqrt (a * x ^ 2 + b * x)) (h₂ : { x | 0 ≤ f x } = f '' { x | 0 ≤ f x }) :
  a = 0 ∨ a = -4 := by
  first
  | solve | simp only [h₁] at *; ring
  | solve | simp only [h₁] at *; norm_num
  | solve | simp only [h₁] at *; omega
  | solve | simp only [h₁] at *; linarith
  | solve | simp only [h₁] at *; nlinarith
  | solve | left; omega
  | solve | left; norm_num
  | solve | left; nlinarith [h₀, h₁, h₂]
  | solve | right; omega
  | solve | right; norm_num
  | solve | right; nlinarith [h₀, h₁, h₂]
  | solve | linarith [h₀, h₁, h₂]
  | solve | nlinarith [h₀, h₁, h₂]
  | solve | nlinarith [sq_nonneg _, h₀, h₁, h₂]
  | solve | linarith
  | solve | nlinarith
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | decide
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num