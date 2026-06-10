import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_202 : (19 ^ 19 + 99 ^ 99) % 10 = 8 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith
    | simp_all [*]
    | decide