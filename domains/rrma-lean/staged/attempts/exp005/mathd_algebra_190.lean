import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_190 : ((3 : ℝ) / 8 + 7 / 8) / (4 / 5) = 25 / 16 := by
  first
    | norm_num
    | field_simp; ring
    | field_simp; linarith
    | field_simp; norm_num
    | ring
    | omega
    | linarith
    | simp_all
    | decide