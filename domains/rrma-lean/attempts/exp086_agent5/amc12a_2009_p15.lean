import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2009_p15 (n : ℕ) (h₀ : 0 < n)
  (h₁ : (∑ k ∈ Finset.Icc 1 n, ↑k * Complex.I ^ k) = 48 + 49 * Complex.I) : n = 97 := by
  first
  | solve | native_decide
  | solve | ring
  | solve | norm_num
  | solve | simp only [h₀, h₁]; ring
  | solve | linarith [h₀, h₁]
  | solve | nlinarith [h₀, h₁]
  | solve | nlinarith [sq_nonneg _, h₀, h₁]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀, h₁]; norm_num
  | solve | simp only [h₀, h₁]; omega
  | solve | simp only [h₀, h₁]; linarith
  | solve | simp only [h₀, h₁]; nlinarith
  | solve | decide
  | solve | simp; norm_num
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | simp; ring
  | solve | simp; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num