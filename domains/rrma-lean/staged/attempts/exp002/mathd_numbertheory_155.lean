import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_155 :
  Finset.card (Finset.filter (fun x => x % 19 = 7) (Finset.Icc 100 999)) = 48 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith
    | simp_all [*]
    | decide