import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_690 :
    IsLeast { a : ℕ | 0 < a ∧ a ≡ 2 [MOD 3] ∧ a ≡ 4 [MOD 5] ∧ a ≡ 6 [MOD 7] ∧ a ≡ 8 [MOD 9] } 314 := by
  first
  | solve | constructor <;> omega
  | solve | constructor <;> norm_num
  | solve | constructor <;> ring
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | decide
  | solve | simp
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | simp; linarith
  | solve | norm_num; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num