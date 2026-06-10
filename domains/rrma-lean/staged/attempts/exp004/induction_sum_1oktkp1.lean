import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_sum_1oktkp1 (n : ℕ) :
  (∑ k ∈ Finset.range n, (1 : ℝ) / ((k + 1) * (k + 2))) = n / (n + 1) := by
  first
    | norm_num
    | native_decide
    | ring
    | omega
    | linarith
    | simp_all
    | decide