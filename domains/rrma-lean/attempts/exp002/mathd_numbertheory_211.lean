import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_211 :
  Finset.card (Finset.filter (fun n => 6 ∣ 4 * ↑n - (2 : ℤ)) (Finset.range 60)) = 20 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg n]
    | simp_all [*]
    | decide