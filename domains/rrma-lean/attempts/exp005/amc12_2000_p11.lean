import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12_2000_p11 (a b : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : a * b = a - b) :
    a / b + b / a - a * b = 2 := by
  first
    | field_simp; linarith [h₀, h₁]
    | field_simp; nlinarith [h₀, h₁]
    | field_simp; ring
    | field_simp; linarith
    | field_simp; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide