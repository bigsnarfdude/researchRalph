import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aime_1990_p2 :
  (52 + 6 * Real.sqrt 43) ^ ((3 : ℝ) / 2) - (52 - 6 * Real.sqrt 43) ^ ((3 : ℝ) / 2) = 828 := by
  first
    | norm_num
    | native_decide
    | field_simp; ring
    | ring
    | omega
    | linarith
    | simp_all
    | decide