import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_149 :
  (∑ k ∈ Finset.filter (fun x => x % 8 = 5 ∧ x % 6 = 3) (Finset.range 50), k) = 66 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith
    | simp_all
    | decide)