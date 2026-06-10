import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_140 (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : ∀ x, 24 * x ^ 2 - 19 * x - 35 = (a * x - 5) * (2 * (b * x) + c)) : a * b - 3 * c = -9 := by
  try
    constructor
        · linear_combination h₀
        · linear_combination h₁
  try
    constructor
        · linear_combination 2 * h₀ - h₁
        · linear_combination h₁ - h₀
  first
  | solve | simp only [h₁] at *; ring
  | solve | simp only [h₁] at *; norm_num
  | solve | simp only [h₁] at *; omega
  | solve | simp only [h₁] at *; linarith
  | solve | simp only [h₁] at *; nlinarith
  | solve | simp only [h₁] at *; constructor <;> (first | norm_num | omega | linarith | nlinarith)
  | solve | linarith [h₀, h₁]
  | solve | nlinarith [h₀, h₁]
  | solve | nlinarith [sq_nonneg _, h₀, h₁]
  | solve | linarith
  | solve | nlinarith
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
  | solve | decide
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num