import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2008_p2 (x : ℝ) (h₀ : x * (1 / 2 + 2 / 3) = 1) : x = 6 / 7 := by
  first
    | field_simp; linarith [h₀]
    | field_simp; nlinarith [h₀]
    | field_simp; ring
    | field_simp; linarith
    | field_simp; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide