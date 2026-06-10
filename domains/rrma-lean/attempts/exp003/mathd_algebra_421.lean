import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_421 (a b c d : ℝ) (h₀ : b = a ^ 2 + 4 * a + 6)
  (h₁ : b = 1 / 2 * a ^ 2 + a + 6) (h₂ : d = c ^ 2 + 4 * c + 6) (h₃ : d = 1 / 2 * c ^ 2 + c + 6)
  (h₄ : a < c) : c - a = 6 := by
  first
  | solve | nlinarith [sq_nonneg (a - b), sq_nonneg a, sq_nonneg b]
  | solve | nlinarith [sq_nonneg (_ - _)]
  | solve | linarith
  | solve | nlinarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₁, h₂, h₃, h₄]
  | solve | nlinarith [h₀, h₁, h₂, h₃, h₄]
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