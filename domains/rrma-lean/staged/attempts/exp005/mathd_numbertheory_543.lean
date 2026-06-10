import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_543 : (∑ k ∈ Nat.divisors (30 ^ 4), 1) - 2 = 123 := by
  first
    | norm_num
    | native_decide
    | decide
    | ring
    | omega
    | linarith
    | simp_all