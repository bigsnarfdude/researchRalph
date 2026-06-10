import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1984_p2 (a b : ℤ) (h₀ : 0 < a ∧ 0 < b) (h₁ : ¬7 ∣ a) (h₂ : ¬7 ∣ b) (h₃ : ¬7 ∣ a + b)
  (h₄ : 7 ^ 7 ∣ (a + b) ^ 7 - a ^ 7 - b ^ 7) : 19 ≤ a + b := by
  try
    constructor
        · linear_combination h₀
        · linear_combination h₁
  try
    constructor
        · linear_combination 2 * h₀ - h₁
        · linear_combination h₁ - h₀
  first
  | solve | constructor <;> linarith [h₀, h₁, h₂, h₃, h₄]
  | solve | constructor <;> nlinarith [h₀, h₁, h₂, h₃, h₄]
  | solve | constructor <;> omega
  | solve | constructor <;> nlinarith [sq_nonneg _, h₀, h₁, h₂, h₃, h₄]
  | solve | linarith [h₀, h₁, h₂, h₃, h₄]
  | solve | nlinarith [h₀, h₁, h₂, h₃, h₄]
  | solve | nlinarith [sq_nonneg _, h₀, h₁, h₂, h₃, h₄]
  | solve | linarith
  | solve | nlinarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; omega
  | solve | omega
  | solve | norm_num
  | solve | simp; omega
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; nlinarith
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
  | solve | ring
  | solve | decide
  | solve | simp; ring
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num