import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem induction_sum2kp1npqsqm1 (n : ℕ) :
  ∑ k ∈ Finset.range n, (2 * k + 3) = (n + 1) ^ 2 - 1 := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ, ih]
    have h1 : 1 ≤ (n + 1) ^ 2 := Nat.one_le_pow 2 (n + 1) (by omega)
    have h2 : 1 ≤ (n + 2) ^ 2 := Nat.one_le_pow 2 (n + 2) (by omega)
    omega
