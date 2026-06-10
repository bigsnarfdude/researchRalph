import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_divisibility_3divnto3m2n (n : ℕ) : 3 ∣ n ^ 3 + 2 * n := by
  induction n with
  | zero => simp
  | succ k ih => ring_nf; omega