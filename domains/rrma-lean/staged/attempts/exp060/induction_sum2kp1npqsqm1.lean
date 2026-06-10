import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem induction_sum2kp1npqsqm1 (n : ℕ) :
  ∑ k ∈ Finset.range n, (2 * k + 3) = (n + 1) ^ 2 - 1 := by
  suffices h : ∑ k ∈ Finset.range n, (2 * k + 3) + 1 = (n + 1) ^ 2 by omega
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ]
    ring_nf
    ring_nf at ih
    omega
