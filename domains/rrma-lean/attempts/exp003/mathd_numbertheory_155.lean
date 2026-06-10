import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_155 :
  Finset.card (Finset.filter (fun x => x % 19 = 7) (Finset.Icc 100 999)) = 48 := by
  first
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