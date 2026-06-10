import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_amgm_prod1toneq1_sum1tongeqn (a : ℕ → NNReal) (n : ℕ)
  (h₀ : Finset.prod (Finset.range n) a = 1) : Finset.sum (Finset.range n) a ≥ n := by
  first
    | omega
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | ring
    | norm_num
    | linarith
    | simp_all