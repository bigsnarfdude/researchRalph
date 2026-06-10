import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem imo_1990_p3 (n : ℕ) (h₀ : 2 ≤ n) (h₁ : n ^ 2 ∣ 2 ^ n + 1) : n = 3 := by
  first
  | solve | nlinarith [sq_nonneg (a - b), sq_nonneg a, sq_nonneg b]
  | solve | nlinarith [sq_nonneg (_ - _)]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀, h₁]
  | solve | simp only [h₀, h₁]; ring
  | solve | simp only [h₀, h₁]; norm_num
  | solve | simp only [h₀, h₁]; linarith
  | solve | simp only [h₀, h₁]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₁]
  | solve | nlinarith [h₀, h₁]
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