import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_536 : ↑3! * ((2 : ℝ) ^ 3 + Real.sqrt 9) / 2 = (33 : ℝ) := by
  first
    | norm_num
    | native_decide
    | ring
    | omega
    | linarith
    | simp_all
    | decide