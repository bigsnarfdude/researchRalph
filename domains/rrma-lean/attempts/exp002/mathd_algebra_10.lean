import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_10 : abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10 := by
  first
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith
    | simp_all [*]
    | decide