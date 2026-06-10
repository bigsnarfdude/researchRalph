import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem imo_1973_p3 (a b : ℝ) (h₀ : ∃ x, x ^ 4 + a * x ^ 3 + b * x ^ 2 + a * x + 1 = 0) :
  4 / 5 ≤ a ^ 2 + b ^ 2 := by
  first
  | solve | exact ⟨0, by omega⟩
  | solve | exact ⟨0, by norm_num⟩
  | solve | exact ⟨1, by omega⟩
  | solve | exact ⟨1, by norm_num⟩
  | solve | exact ⟨2, by omega⟩
  | solve | exact ⟨2, by norm_num⟩
  | solve | exact ⟨3, by omega⟩
  | solve | exact ⟨3, by norm_num⟩
  | solve | exact ⟨4, by omega⟩
  | solve | exact ⟨4, by norm_num⟩
  | solve | exact ⟨5, by omega⟩
  | solve | exact ⟨5, by norm_num⟩
  | solve | nlinarith [sq_nonneg (a - b), sq_nonneg a, sq_nonneg b]
  | solve | nlinarith [sq_nonneg (_ - _)]
  | solve | linarith
  | solve | nlinarith
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