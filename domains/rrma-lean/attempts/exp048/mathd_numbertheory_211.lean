import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_211 :
  Finset.card (Finset.filter (fun n => 6 ∣ 4 * ↑n - (2 : ℤ)) (Finset.range 60)) = 20 := by
  first
  | solve | native_decide
  | solve | norm_num
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp; omega
  | solve | decide
  | solve | simp; norm_num
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | ring
  | solve | simp; ring
  | solve | push_cast; ring
  | solve | push_cast; norm_num