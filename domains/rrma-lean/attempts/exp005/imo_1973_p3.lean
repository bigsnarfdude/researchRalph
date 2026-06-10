import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1973_p3 (a b : ℝ) (h₀ : ∃ x, x ^ 4 + a * x ^ 3 + b * x ^ 2 + a * x + 1 = 0) :
  4 / 5 ≤ a ^ 2 + b ^ 2 := by
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