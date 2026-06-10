import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2020_p13 (a b c : ℕ) (n : NNReal) (h₀ : n ≠ 1) (h₁ : 1 < a ∧ 1 < b ∧ 1 < c)
  (h₂ : (n * (n * n ^ (1 / c)) ^ (1 / b)) ^ (1 / a) = (n ^ 25) ^ (1 / 36)) : b = 3 := by
  try
    constructor
        · linear_combination h₀
        · linear_combination h₁
  try
    constructor
        · linear_combination 2 * h₀ - h₁
        · linear_combination h₁ - h₀
  first
  | solve | constructor <;> linarith [h₀, h₁, h₂]
  | solve | constructor <;> nlinarith [h₀, h₁, h₂]
  | solve | constructor <;> omega
  | solve | constructor <;> nlinarith [sq_nonneg _, h₀, h₁, h₂]
  | solve | field_simp; linarith [h₀, h₁, h₂]
  | solve | field_simp; nlinarith [h₀, h₁, h₂]
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; norm_num
  | solve | linarith [h₀, h₁, h₂]
  | solve | nlinarith [h₀, h₁, h₂]
  | solve | nlinarith [sq_nonneg _, h₀, h₁, h₂]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀, h₁, h₂]; ring
  | solve | simp only [h₀, h₁, h₂]; norm_num
  | solve | simp only [h₀, h₁, h₂]; omega
  | solve | simp only [h₀, h₁, h₂]; linarith
  | solve | simp only [h₀, h₁, h₂]; nlinarith
  | solve | linear_combination h₀
  | solve | linear_combination h₁
  | solve | linear_combination h₂
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
  | solve | norm_num
  | solve | ring
  | solve | decide
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num