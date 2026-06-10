import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem algebra_amgm_prod1toneq1_sum1tongeqn (a : ℕ → NNReal) (n : ℕ)
  (h₀ : Finset.prod (Finset.range n) a = 1) : Finset.sum (Finset.range n) a ≥ n := by
  first
  | solve | linarith
  | solve | nlinarith
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