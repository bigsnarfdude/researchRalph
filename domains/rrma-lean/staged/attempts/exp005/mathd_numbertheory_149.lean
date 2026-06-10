import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_149 :
  (∑ k ∈ Finset.filter (fun x => x % 8 = 5 ∧ x % 6 = 3) (Finset.range 50), k) = 66 := by
  first
    | omega
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | norm_num
    | constructor <;> norm_num
    | constructor <;> simp_all
    | ring
    | linarith
    | simp_all