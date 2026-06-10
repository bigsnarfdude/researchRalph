import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem algebra_amgm_sumasqdivbsqgeqsumbdiva (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c) :
  a ^ 2 / b ^ 2 + b ^ 2 / c ^ 2 + c ^ 2 / a ^ 2 ≥ b / a + c / b + a / c := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg c, sq_nonneg h₀, sq_nonneg (a - b), sq_nonneg (a + b), mul_self_nonneg (a - b)]
    | simp_all
    | decide)