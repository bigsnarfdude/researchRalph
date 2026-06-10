import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_67 (f g : ℝ → ℝ) (h₀ : ∀ x, f x = 5 * x + 3) (h₁ : ∀ x, g x = x ^ 2 - 2) :
    g (f (-1)) = 2 := by
  first
  | solve | simp only [h₀, h₁] at *; ring
  | solve | simp only [h₀, h₁] at *; norm_num
  | solve | simp only [h₀, h₁] at *; omega
  | solve | simp only [h₀, h₁] at *; linarith
  | solve | simp only [h₀, h₁] at *; nlinarith
  | solve | linear_combination h₀
  | solve | linear_combination h₁
  | solve | linear_combination h₀ + h₁
  | solve | linear_combination h₀ - h₁
  | solve | linear_combination 2 * h₀ - h₁
  | solve | linear_combination h₁ - h₀
  | solve | linear_combination 2 * h₁ - h₀
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | decide
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num