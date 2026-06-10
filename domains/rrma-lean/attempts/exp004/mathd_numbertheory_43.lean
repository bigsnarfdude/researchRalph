import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_43 : IsGreatest { n : ℕ | 15 ^ n ∣ 942! } 233 := by
  first
    | omega
    | norm_num
    | native_decide
    | ring
    | linarith
    | simp_all
    | decide