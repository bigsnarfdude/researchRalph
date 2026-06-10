import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_690 :
    IsLeast { a : ℕ | 0 < a ∧ a ≡ 2 [MOD 3] ∧ a ≡ 4 [MOD 5] ∧ a ≡ 6 [MOD 7] ∧ a ≡ 8 [MOD 9] } 314 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith
    | simp_all
    | decide)