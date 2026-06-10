import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_169 : Nat.gcd 20! 200000 = 40000 := by
  norm_num