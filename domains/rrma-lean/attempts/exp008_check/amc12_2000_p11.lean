import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12_2000_p11 (a b : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : a * b = a - b) :
    a / b + b / a - a * b = 2 := by
  try
    constructor
        · linear_combination h₀
        · linear_combination h₁
  try
    constructor
        · linear_combination 2 * h₀ - h₁
        · linear_combination h₁ - h₀
  first
  | solve | constructor <;> linarith [h₀, h₁]
  | solve | constructor <;> nlinarith [h₀, h₁]
  | solve | constructor <;> omega
  | solve | field_simp; linarith [h₀, h₁]
  | solve | field_simp; nlinarith [h₀, h₁]
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; norm_num
  | solve | simp only [h₀, h₁]; ring
  | solve | simp only [h₀, h₁]; norm_num
  | solve | simp only [h₀, h₁]; omega
  | solve | simp only [h₀, h₁]; linarith
  | solve | simp only [h₀, h₁]; nlinarith
  | solve | linarith [h₀, h₁]
  | solve | nlinarith [h₀, h₁]
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