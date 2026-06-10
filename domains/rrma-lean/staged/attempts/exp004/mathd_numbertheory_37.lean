import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_37 : Nat.lcm 9999 100001 = 90900909 := by
  first
    | norm_num
    | native_decide
    | ring
    | omega
    | linarith
    | simp_all
    | decide