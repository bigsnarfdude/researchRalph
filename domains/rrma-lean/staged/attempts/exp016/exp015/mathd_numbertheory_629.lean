import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_629 : IsLeast { t : ℕ | 0 < t ∧ Nat.lcm 12 t ^ 3 = (12 * t) ^ 2 } 18 := by
  first
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | simp
  | solve | simp_all
  | solve | decide
  | solve | native_decide
  | solve | push_cast; ring
  | solve | push_cast; norm_num
  | solve | push_cast; omega
  | solve | field_simp; ring
  | solve | field_simp; norm_num
  | solve | ring_nf; norm_num
  | solve | ring_nf; omega
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; norm_num
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith