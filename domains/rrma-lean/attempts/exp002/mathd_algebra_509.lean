import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_509 :
  Real.sqrt ((5 / Real.sqrt 80 + Real.sqrt 845 / 9 + Real.sqrt 45) / Real.sqrt 5) = 13 / 6 := by
  first
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith
    | simp_all [*]
    | decide