import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_102 : 2 ^ 8 % 5 = 1 := by
  first
    | omega
    | norm_num
    | native_decide
    | decide
    | ring
    | linarith
    | simp_all