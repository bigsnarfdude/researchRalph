import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_sum2kp1npqsqm1 (n : ℕ) :
  ∑ k ∈ Finset.range n, (2 * k + 3) = (n + 1) ^ 2 - 1 := by
  first
    | omega
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | norm_num
    | ring
    | linarith
    | simp_all