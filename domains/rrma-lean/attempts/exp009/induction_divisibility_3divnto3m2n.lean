import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem induction_divisibility_3divnto3m2n (n : ℕ) : 3 ∣ n ^ 3 + 2 * n := by
  induction n with
  | zero => norm_num
  | succ n ih =>
    have : (n + 1) ^ 3 + 2 * (n + 1) = (n ^ 3 + 2 * n) + 3 * (n ^ 2 + n + 1) := by ring
    rw [this]
    exact Nat.dvd_add ih (dvd_mul_right 3 _)
