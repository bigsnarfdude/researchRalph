import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aime_1994_p4 (n : ℕ) (h₀ : 0 < n)
  (h₀ : (∑ k ∈ Finset.Icc 1 n, Int.floor (Real.logb 2 k)) = 1994) : n = 312 := by
  first
  | solve | native_decide
  | solve | linarith [h₀, h₀]
  | solve | nlinarith [h₀, h₀]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀, h₀]; ring
  | solve | simp only [h₀, h₀]; norm_num
  | solve | simp only [h₀, h₀]; omega
  | solve | simp only [h₀, h₀]; linarith
  | solve | simp only [h₀, h₀]; nlinarith
  | solve | decide
  | solve | simp; norm_num
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | norm_num
  | solve | ring
  | solve | simp; ring
  | solve | simp; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num