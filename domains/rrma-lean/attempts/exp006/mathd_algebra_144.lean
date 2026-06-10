import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_144 (a b c d : ℕ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h₀ : (c : ℤ) - b = d)
    (h₁ : (b : ℤ) - a = d) (h₂ : a + b + c = 60) (h₃ : a + b > c) : d < 10 := by
  try
    constructor
        · linear_combination h₀
        · linear_combination h₀
  try
    constructor
        · linear_combination 2 * h₀ - h₀
        · linear_combination h₀ - h₀
  first
  | solve | constructor <;> linarith [h₀, h₀, h₁, h₂, h₃]
  | solve | constructor <;> nlinarith [h₀, h₀, h₁, h₂, h₃]
  | solve | constructor <;> omega
  | solve | constructor <;> nlinarith [sq_nonneg _, h₀, h₀, h₁, h₂, h₃]
  | solve | linarith [h₀, h₀, h₁, h₂, h₃]
  | solve | nlinarith [h₀, h₀, h₁, h₂, h₃]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]; ring
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]; norm_num
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]; omega
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]; linarith
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]; nlinarith
  | solve | linear_combination h₀
  | solve | linear_combination h₁
  | solve | linear_combination h₂
  | solve | linear_combination h₃
  | solve | linear_combination h₀ + h₀
  | solve | linear_combination h₀ - h₀
  | solve | linear_combination 2 * h₀ - h₀
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