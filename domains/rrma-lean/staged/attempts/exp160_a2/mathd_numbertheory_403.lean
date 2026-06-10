import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_numbertheory_403 : (∑ k ∈ Nat.properDivisors 198, k) = 270 := by native_decide
