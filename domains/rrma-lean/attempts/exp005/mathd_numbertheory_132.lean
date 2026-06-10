import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_132 : 2004 % 12 = 0 := by
  first
    | omega
    | norm_num
    | native_decide
    | decide
    | ring
    | linarith
    | simp_all