import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_104 (x : ℝ) (h₀ : 125 / 8 = x / 12) : x = 375 / 2 := by
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