import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem induction_divisibility_3div2tooddnp1 (n : ℕ) : 3 ∣ 2 ^ (2 * n + 1) + 1 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg n]
    | simp_all [*]
    | decide