import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_22 : Real.logb (5 ^ 2) (5 ^ 4) = 2 := by
  first
    | norm_num
    | native_decide
    | decide
    | ring
    | omega
    | linarith
    | simp_all