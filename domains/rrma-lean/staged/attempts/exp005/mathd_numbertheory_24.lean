import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_24 : (∑ k ∈ Finset.Icc 1 9, 11 ^ k) % 100 = 59 := by
  first
    | omega
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | norm_num
    | ring
    | linarith
    | simp_all