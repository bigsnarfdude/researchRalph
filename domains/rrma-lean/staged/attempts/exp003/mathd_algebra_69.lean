import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_69 (rows seats : ℕ) (h₀ : rows * seats = 450)
  (h₁ : (rows + 5) * (seats - 3) = 450) : rows = 25 := by
  first
  | solve | ring
  | solve | norm_num
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