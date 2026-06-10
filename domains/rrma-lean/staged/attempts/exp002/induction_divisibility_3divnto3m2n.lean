import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem induction_divisibility_3divnto3m2n (n : ℕ) : 3 ∣ n ^ 3 + 2 * n := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg n]
    | simp_all [*]
    | decide