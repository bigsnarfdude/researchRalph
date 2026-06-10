import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_amgm_sumasqdivbsqgeqsumbdiva (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c) :
  a ^ 2 / b ^ 2 + b ^ 2 / c ^ 2 + c ^ 2 / a ^ 2 ≥ b / a + c / b + a / c := by
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