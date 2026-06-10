import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem algebra_amgm_prod1toneq1_sum1tongeqn (a : ℕ → NNReal) (n : ℕ)
  (h₀ : Finset.prod (Finset.range n) a = 1) : Finset.sum (Finset.range n) a ≥ n := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg n, sq_nonneg h₀, sq_nonneg (a - n), sq_nonneg (a + n), mul_self_nonneg (a - n)]
    | simp_all [*]
    | decide