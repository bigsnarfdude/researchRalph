import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_sum_odd (n : ℕ) : (∑ k ∈ Finset.range n, (2 * k + 1)) = n ^ 2 := by
  first
    | omega
    | norm_num
    | native_decide
    | ring
    | linarith
    | simp_all
    | decide