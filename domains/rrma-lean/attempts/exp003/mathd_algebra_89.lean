import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_89 (b : ℝ) (h₀ : b ≠ 0) :
  (7 * b ^ 3) ^ 2 * (4 * b ^ 2) ^ (-(3 : ℤ)) = 49 / 64 := by
  first
  | solve | ring
  | solve | norm_num
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; nlinarith
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
  | solve | omega
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