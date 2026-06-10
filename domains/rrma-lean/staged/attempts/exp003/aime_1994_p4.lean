import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem aime_1994_p4 (n : ℕ) (h₀ : 0 < n)
  (h₀ : (∑ k ∈ Finset.Icc 1 n, Int.floor (Real.logb 2 k)) = 1994) : n = 312 := by
  first
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀, h₀]
  | solve | simp only [h₀, h₀]; ring
  | solve | simp only [h₀, h₀]; norm_num
  | solve | simp only [h₀, h₀]; linarith
  | solve | simp only [h₀, h₀]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₀]
  | solve | nlinarith [h₀, h₀]
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