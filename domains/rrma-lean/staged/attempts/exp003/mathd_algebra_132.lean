import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_132 (x : ℝ) (f g : ℝ → ℝ) (h₀ : ∀ x, f x = x + 2) (h₁ : ∀ x, g x = x ^ 2)
  (h₂ : f (g x) = g (f x)) : x = -1 / 2 := by
  first
  | solve | ring
  | solve | norm_num
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; nlinarith
  | solve | simp only [h₀, h₁, h₂]
  | solve | simp only [h₀, h₁, h₂]; ring
  | solve | simp only [h₀, h₁, h₂]; norm_num
  | solve | simp only [h₀, h₁, h₂]; linarith
  | solve | simp only [h₀, h₁, h₂]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₁, h₂]
  | solve | nlinarith [h₀, h₁, h₂]
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