import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_96 (x y z a : ℝ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z)
  (h₁ : Real.log x - Real.log y = a) (h₂ : Real.log y - Real.log z = 15)
  (h₃ : Real.log z - Real.log x = -7) : a = -8 := by
  try
    constructor
        · linear_combination h₀
        · linear_combination h₁
  try
    constructor
        · linear_combination 2 * h₀ - h₁
        · linear_combination h₁ - h₀
  first
  | solve | constructor <;> linarith [h₀, h₁, h₂, h₃]
  | solve | constructor <;> nlinarith [h₀, h₁, h₂, h₃]
  | solve | constructor <;> omega
  | solve | constructor <;> nlinarith [sq_nonneg _, h₀, h₁, h₂, h₃]
  | solve | linarith [h₀, h₁, h₂, h₃]
  | solve | nlinarith [h₀, h₁, h₂, h₃]
  | solve | linarith
  | solve | nlinarith
  | solve | simp only [h₀, h₁, h₂, h₃]; ring
  | solve | simp only [h₀, h₁, h₂, h₃]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃]; omega
  | solve | simp only [h₀, h₁, h₂, h₃]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃]; nlinarith
  | solve | linear_combination h₀
  | solve | linear_combination h₁
  | solve | linear_combination h₂
  | solve | linear_combination h₃
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