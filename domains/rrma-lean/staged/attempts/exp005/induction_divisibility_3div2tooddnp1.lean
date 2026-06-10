import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_divisibility_3div2tooddnp1 (n : ℕ) : 3 ∣ 2 ^ (2 * n + 1) + 1 := by
  first
    | omega
    | norm_num
    | native_decide
    | decide
    | ring
    | linarith
    | simp_all