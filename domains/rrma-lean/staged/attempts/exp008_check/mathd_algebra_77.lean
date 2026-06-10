import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_77 (a b : ℝ) (f : ℝ → ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : a ≠ b)
  (h₂ : ∀ x, f x = x ^ 2 + a * x + b) (h₃ : f a = 0) (h₄ : f b = 0) : a = 1 ∧ b = -2 := by
  try
    constructor
        · linear_combination h₀
        · linear_combination h₁
  try
    constructor
        · linear_combination 2 * h₀ - h₁
        · linear_combination h₁ - h₀
  first
  | solve | simp only [h₂] at *; ring
  | solve | simp only [h₂] at *; norm_num
  | solve | simp only [h₂] at *; omega
  | solve | simp only [h₂] at *; linarith
  | solve | simp only [h₂] at *; nlinarith
  | solve | simp only [h₂] at *; constructor <;> (first | norm_num | omega | linarith | nlinarith)
  | solve | linear_combination h₀
  | solve | linear_combination h₁
  | solve | linear_combination h₂
  | solve | linear_combination h₃
  | solve | linear_combination h₄
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