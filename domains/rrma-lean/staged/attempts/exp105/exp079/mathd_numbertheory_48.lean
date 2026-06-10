import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_48 (b : ℕ) (h₀ : 0 < b) (h₁ : 3 * b ^ 2 + 2 * b + 1 = 57) : b = 4 := by
  first
  | solve | linarith [h₀, h₁]
  | solve | nlinarith [h₀, h₁]
  | solve | nlinarith [sq_nonneg _, h₀, h₁]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀, h₁]; ring
  | solve | simp only [h₀, h₁]; norm_num
  | solve | simp only [h₀, h₁]; omega
  | solve | simp only [h₀, h₁]; linarith
  | solve | simp only [h₀, h₁]; nlinarith
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