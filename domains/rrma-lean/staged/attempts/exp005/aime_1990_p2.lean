import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aime_1990_p2 :
  (52 + 6 * Real.sqrt 43) ^ ((3 : ℝ) / 2) - (52 - 6 * Real.sqrt 43) ^ ((3 : ℝ) / 2) = 828 := by
  first
    | norm_num
    | field_simp; ring
    | field_simp; linarith
    | field_simp; norm_num
    | ring
    | omega
    | linarith
    | simp_all
    | decide