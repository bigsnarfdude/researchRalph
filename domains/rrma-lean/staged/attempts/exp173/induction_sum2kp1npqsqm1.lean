import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem induction_sum2kp1npqsqm1 (n : ℕ) :
  ∑ k ∈ Finset.range n, (2 * k + 3) = (n + 1) ^ 2 - 1 := by
  have : ∑ k ∈ Finset.range n, (2 * k + 3) = n ^ 2 + 2 * n := by
    induction n with
    | zero => simp
    | succ k ih =>
      rw [Finset.sum_range_succ, ih]
      ring
  rw [this]
  have : (n + 1) ^ 2 = n ^ 2 + 2 * n + 1 := by ring
  omega
