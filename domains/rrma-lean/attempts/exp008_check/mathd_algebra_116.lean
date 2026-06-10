import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_116 (k x : ℝ) (h₀ : x = (13 - Real.sqrt 131) / 4)
    (h₁ : 2 * x ^ 2 - 13 * x + k = 0) : k = 19 / 4 := by
  first
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | field_simp; linarith [h₀, h₁]
  | solve | field_simp; nlinarith [h₀, h₁]
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; norm_num
  | solve | linear_combination h₀
  | solve | linear_combination h₁
  | solve | linear_combination h₀ + h₁
  | solve | linear_combination h₀ - h₁
  | solve | linear_combination 2 * h₀ - h₁
  | solve | linear_combination h₁ - h₀
  | solve | linear_combination 2 * h₁ - h₀
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | decide
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num