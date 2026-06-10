import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aime_1983_p9 (x : ℝ) (h₀ : 0 < x ∧ x < Real.pi) :
  12 ≤ (9 * (x ^ 2 * Real.sin x ^ 2) + 4) / (x * Real.sin x) := by
  first
  | solve | constructor <;> linarith [h₀]
  | solve | constructor <;> nlinarith [h₀]
  | solve | constructor <;> omega
  | solve | constructor <;> nlinarith [sq_nonneg _, h₀]
  | solve | field_simp; linarith [h₀]
  | solve | field_simp; nlinarith [h₀]
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; norm_num
  | solve | linarith [h₀]
  | solve | nlinarith [h₀]
  | solve | nlinarith [sq_nonneg _, h₀]
  | solve | linarith
  | solve | nlinarith
  | solve | simp only [h₀]; ring
  | solve | simp only [h₀]; norm_num
  | solve | simp only [h₀]; omega
  | solve | simp only [h₀]; linarith
  | solve | simp only [h₀]; nlinarith
  | solve | linear_combination h₀
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | decide
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num