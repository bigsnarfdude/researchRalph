import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_amgm_prod1toneq1_sum1tongeqn (a : ℕ → NNReal) (n : ℕ)
  (h₀ : Finset.prod (Finset.range n) a = 1) : Finset.sum (Finset.range n) a ≥ n := by
  first
  | solve | native_decide
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