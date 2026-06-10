import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem induction_divisibility_3divnto3m2n (n : ℕ) : 3 ∣ n ^ 3 + 2 * n := by
  induction n with
  | zero => norm_num
  | succ k ih =>
    have : (k + 1) ^ 3 + 2 * (k + 1) = k ^ 3 + 2 * k + 3 * (k ^ 2 + k + 1) := by ring
    rw [this]
    exact Nat.dvd_add ih (Nat.dvd_mul_right 3 _)
