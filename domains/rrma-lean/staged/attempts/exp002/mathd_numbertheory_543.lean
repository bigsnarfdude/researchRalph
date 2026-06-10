import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_543 : (∑ k ∈ Nat.divisors (30 ^ 4), 1) - 2 = 123 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith
    | simp_all [*]
    | decide