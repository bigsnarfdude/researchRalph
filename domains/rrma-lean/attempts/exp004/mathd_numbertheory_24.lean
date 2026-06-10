import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_24 : (∑ k ∈ Finset.Icc 1 9, 11 ^ k) % 100 = 59 := by
  first
    | omega
    | norm_num
    | native_decide
    | ring
    | linarith
    | simp_all
    | decide