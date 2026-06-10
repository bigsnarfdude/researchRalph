import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2017_p2 (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : x + y = 4 * (x * y)) :
  1 / x + 1 / y = 4 := by
  first
    | field_simp; linarith [h₀, h₁, h₂]
    | field_simp; nlinarith [h₀, h₁, h₂]
    | field_simp; ring
    | field_simp; linarith
    | field_simp; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide