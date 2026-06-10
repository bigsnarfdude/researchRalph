import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem induction_sum_odd (n : ℕ) : (∑ k ∈ Finset.range n, (2 * k + 1)) = n ^ 2 := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [Finset.sum_range_succ, ih]
    ring
