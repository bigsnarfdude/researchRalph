import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem induction_divisibility_3div2tooddnp1 (n : ℕ) : 3 ∣ 2 ^ (2 * n + 1) + 1 := by
  induction n with
  | zero => norm_num
  | succ k ih =>
    rw [show 2 * (k + 1) + 1 = 2 * k + 1 + 2 from by ring]
    rw [pow_add]
    omega
