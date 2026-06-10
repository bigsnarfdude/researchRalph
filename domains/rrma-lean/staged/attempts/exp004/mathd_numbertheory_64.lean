import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_64 : IsLeast { x : ℕ | 30 * x ≡ 42 [MOD 47] } 39 := by
  first
    | omega
    | norm_num
    | native_decide
    | ring
    | linarith
    | simp_all
    | decide