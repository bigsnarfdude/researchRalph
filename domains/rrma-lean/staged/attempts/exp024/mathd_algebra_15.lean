import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_15 (s : ℕ → ℕ → ℕ)
    (h₀ : ∀ a b, 0 < a ∧ 0 < b → s a b = a ^ (b : ℕ) + b ^ (a : ℕ)) : s 2 6 = 100 := by
  first
  | solve | simp only [h₀] at *; ring
  | solve | simp only [h₀] at *; norm_num
  | solve | simp only [h₀] at *; omega
  | solve | simp only [h₀] at *; linarith
  | solve | simp only [h₀] at *; nlinarith
  | solve | simp only [h₀] at *; constructor <;> (first | norm_num | omega | linarith | nlinarith)
  | solve | linarith [h₀]
  | solve | nlinarith [h₀]
  | solve | nlinarith [sq_nonneg _, h₀]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | linear_combination h₀
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