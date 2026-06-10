import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2009_p2 : 1 + 1 / (1 + 1 / (1 + 1)) = (5 : ℚ) / 3 := by
  first
    | norm_num
    | native_decide
    | decide
    | field_simp; ring
    | field_simp; linarith
    | field_simp; norm_num
    | ring
    | omega
    | linarith
    | simp_all