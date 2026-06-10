import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_149 :
  (∑ k ∈ Finset.filter (fun x => x % 8 = 5 ∧ x % 6 = 3) (Finset.range 50), k) = 66 := by
  first
  | solve | native_decide
  | solve | norm_num
  | solve | constructor <;> omega
  | solve | constructor <;> norm_num
  | solve | linarith
  | solve | nlinarith
  | solve | decide
  | solve | simp; norm_num
  | solve | omega
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | ring
  | solve | simp; ring
  | solve | simp; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num