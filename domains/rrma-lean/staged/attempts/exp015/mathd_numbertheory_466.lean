import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_466 : (∑ k ∈ Finset.range 11, k) % 9 = 1 := by
  norm_num