import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_numbertheory_202 : (19 ^ 19 + 99 ^ 99) % 10 = 8 := by native_decide
