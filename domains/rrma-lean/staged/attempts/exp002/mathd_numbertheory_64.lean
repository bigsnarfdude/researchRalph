import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_64 : IsLeast { x : ℕ | 30 * x ≡ 42 [MOD 47] } 39 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith
    | simp_all [*]
    | decide