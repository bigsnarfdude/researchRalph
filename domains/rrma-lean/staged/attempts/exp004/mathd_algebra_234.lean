import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_234 (d : ℝ) (h₀ : 27 / 125 * d = 9 / 25) : 3 / 5 * d ^ 3 = 25 / 9 := by
  first
    | field_simp; linarith [h₀]
    | field_simp; ring
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide