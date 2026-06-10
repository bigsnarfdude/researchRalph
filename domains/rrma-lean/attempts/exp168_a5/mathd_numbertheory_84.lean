import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_84 : Int.floor ((9 : ℝ) / 160 * 100) = 5 := by
  norm_num [Int.floor_eq_iff]
