import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_690 :
    IsLeast { a : ℕ | 0 < a ∧ a ≡ 2 [MOD 3] ∧ a ≡ 4 [MOD 5] ∧ a ≡ 6 [MOD 7] ∧ a ≡ 8 [MOD 9] } 314 := by
  first
    | omega
    | norm_num
    | native_decide
    | constructor <;> omega
    | constructor <;> norm_num
    | ring
    | linarith
    | simp_all
    | decide