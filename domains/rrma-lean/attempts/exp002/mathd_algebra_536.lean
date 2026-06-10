import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_536 : ↑3! * ((2 : ℝ) ^ 3 + Real.sqrt 9) / 2 = (33 : ℝ) := by
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