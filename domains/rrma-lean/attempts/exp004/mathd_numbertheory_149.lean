import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_149 :
  (∑ k ∈ Finset.filter (fun x => x % 8 = 5 ∧ x % 6 = 3) (Finset.range 50), k) = 66 := by
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