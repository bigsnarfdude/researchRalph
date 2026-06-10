import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_462 : ((1 : ℚ) / 2 + 1 / 3) * (1 / 2 - 1 / 3) = 5 / 36 := by
  first
    | norm_num
    | native_decide
    | field_simp; ring
    | ring
    | omega
    | linarith
    | simp_all
    | decide