import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_101 : 17 * 18 % 4 = 2 := by
  first
    | omega
    | norm_num
    | native_decide
    | ring
    | linarith
    | simp_all
    | decide