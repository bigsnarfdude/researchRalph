import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_84 : Int.floor ((9 : ℝ) / 160 * 100) = 5 := by
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