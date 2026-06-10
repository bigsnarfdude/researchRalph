import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_301 (j : ℕ) (h₀ : 0 < j) : 3 * (7 * ↑j + 3) % 7 = 2 := by
  first
  | solve | linarith [h₀]
  | solve | nlinarith [h₀]
  | solve | nlinarith [sq_nonneg j, h₀]
  | solve | nlinarith [sq_nonneg (j - 1), h₀]
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | decide
  | solve | native_decide
  | solve | simp only [h₀]; ring
  | solve | simp only [h₀]; norm_num
  | solve | simp only [h₀]; omega
  | solve | simp only [h₀]; linarith
  | solve | simp only [h₀]; nlinarith
  | solve | push_cast; ring
  | solve | push_cast; norm_num
  | solve | push_cast; omega
  | solve | field_simp; ring
  | solve | field_simp; norm_num
  | solve | ring_nf; norm_num
  | solve | ring_nf; omega
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; norm_num
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith