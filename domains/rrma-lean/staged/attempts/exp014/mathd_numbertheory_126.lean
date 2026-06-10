import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_126 (x a : ℕ) (h₀ : 0 < x ∧ 0 < a) (h₁ : Nat.gcd a 40 = x + 3)
  (h₂ : Nat.lcm a 40 = x * (x + 3))
  (h₃ : ∀ b : ℕ, 0 < b → Nat.gcd b 40 = x + 3 ∧ Nat.lcm b 40 = x * (x + 3) → a ≤ b) : a = 8 := by
  try
    constructor
        · linear_combination h₀
        · linear_combination h₁
  try
    constructor
        · linear_combination 2 * h₀ - h₁
        · linear_combination h₁ - h₀
  first
  | solve | norm_num
  | solve | simp only [h₃] at *; ring
  | solve | simp only [h₃] at *; norm_num
  | solve | simp only [h₃] at *; omega
  | solve | simp only [h₃] at *; linarith
  | solve | simp only [h₃] at *; nlinarith
  | solve | simp only [h₃] at *; constructor <;> (first | norm_num | omega | linarith | nlinarith)
  | solve | linarith [h₀, h₁, h₂, h₃]
  | solve | nlinarith [h₀, h₁, h₂, h₃]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | linear_combination h₀
  | solve | linear_combination h₁
  | solve | linear_combination h₂
  | solve | linear_combination h₃
  | solve | linear_combination h₀ + h₁
  | solve | linear_combination h₀ - h₁
  | solve | linear_combination 2 * h₀ - h₁
  | solve | linear_combination h₁ - h₀
  | solve | linear_combination 2 * h₁ - h₀
  | solve | native_decide
  | solve | decide
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | ring
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num