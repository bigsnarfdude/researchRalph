import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_301 (j : ℕ) (h₀ : 0 < j) : 3 * (7 * ↑j + 3) % 7 = 2 := by
  first
  | solve | linarith [h₀]
  | solve | nlinarith [h₀]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀]; ring
  | solve | simp only [h₀]; norm_num
  | solve | simp only [h₀]; omega
  | solve | simp only [h₀]; linarith
  | solve | simp only [h₀]; nlinarith
  | solve | native_decide
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | norm_num
  | solve | ring
  | solve | decide
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num