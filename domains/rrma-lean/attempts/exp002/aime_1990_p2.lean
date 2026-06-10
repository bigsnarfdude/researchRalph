import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem aime_1990_p2 :
  (52 + 6 * Real.sqrt 43) ^ ((3 : ℝ) / 2) - (52 - 6 * Real.sqrt 43) ^ ((3 : ℝ) / 2) = 828 := by
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