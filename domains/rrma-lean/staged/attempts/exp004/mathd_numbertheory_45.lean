import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_45 : Nat.gcd 6432 132 + 11 = 23 := by
  first
    | norm_num
    | native_decide
    | ring
    | omega
    | linarith
    | simp_all
    | decide