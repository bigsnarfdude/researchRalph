import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_sum_odd (n : ℕ) : (∑ k ∈ Finset.range n, (2 * k + 1)) = n ^ 2 := by
  first
    | omega
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | norm_num
    | ring
    | linarith
    | simp_all