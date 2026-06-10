import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_77 (a b : ℝ) (f : ℝ → ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : a ≠ b)
  (h₂ : ∀ x, f x = x ^ 2 + a * x + b) (h₃ : f a = 0) (h₄ : f b = 0) : a = 1 ∧ b = -2 := by
  first
  | solve | constructor <;> linarith
  | solve | constructor <;> nlinarith
  | solve | constructor <;> norm_num
  | solve | constructor <;> ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₁, h₂, h₃, h₄]
  | solve | nlinarith [h₀, h₁, h₂, h₃, h₄]
  | solve | omega
  | solve | norm_num
  | solve | ring
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