import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_780 (m x : ℤ) (h₀ : 0 ≤ x) (h₁ : 10 ≤ m ∧ m ≤ 99) (h₂ : 6 * x % m = 1)
  (h₃ : (x - 6 ^ 2) % m = 0) : m = 43 := by
  first
  | solve | constructor <;> norm_num
  | solve | constructor <;> ring
  | solve | nlinarith [sq_nonneg (a - b), sq_nonneg a, sq_nonneg b]
  | solve | nlinarith [sq_nonneg (_ - _)]
  | solve | linarith
  | solve | nlinarith
  | solve | simp only [h₀, h₁, h₂, h₃]
  | solve | simp only [h₀, h₁, h₂, h₃]; ring
  | solve | simp only [h₀, h₁, h₂, h₃]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₁, h₂, h₃]
  | solve | nlinarith [h₀, h₁, h₂, h₃]
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | decide
  | solve | simp
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | simp; linarith
  | solve | norm_num; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num