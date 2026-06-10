import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_155 :
  Finset.card (Finset.filter (fun x => x % 19 = 7) (Finset.Icc 100 999)) = 48 := by
  native_decide