import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_206 (a b : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = x ^ 2 + a * x + b) (h₁ : 2 * a ≠ b)
  (h₂ : f (2 * a) = 0) (h₃ : f b = 0) : a + b = -1 := by
  first
  | solve | ring
  | solve | norm_num
  | solve | simp only [h₀, h₁, h₂, h₃]
  | solve | simp only [h₀, h₁, h₂, h₃]; ring
  | solve | simp only [h₀, h₁, h₂, h₃]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₁, h₂, h₃]
  | solve | nlinarith [h₀, h₁, h₂, h₃]
  | solve | omega
  | solve | linarith
  | solve | nlinarith
  | solve | decide
  | solve | simp
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | simp; linarith
  | solve | norm_num; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num