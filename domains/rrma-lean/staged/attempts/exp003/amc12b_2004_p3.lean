import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12b_2004_p3 (x y : ℕ) (h₀ : 2 ^ x * 3 ^ y = 1296) : x + y = 8 := by
  first
  | solve | ring
  | solve | norm_num
  | solve | omega
  | solve | simp only [h₀]
  | solve | simp only [h₀]; ring
  | solve | simp only [h₀]; norm_num
  | solve | simp only [h₀]; linarith
  | solve | simp only [h₀]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀]
  | solve | linarith
  | solve | nlinarith
  | solve | decide
  | solve | simp
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | simp; linarith
  | solve | norm_num; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num