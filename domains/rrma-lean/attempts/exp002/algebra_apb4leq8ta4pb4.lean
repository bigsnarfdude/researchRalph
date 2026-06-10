import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem algebra_apb4leq8ta4pb4 (a b : ℝ) (h₀ : 0 < a ∧ 0 < b) : (a + b) ^ 4 ≤ 8 * (a ^ 4 + b ^ 4) := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg h₀, sq_nonneg (a - b), sq_nonneg (a + b), mul_self_nonneg (a - b)]
    | simp_all
    | decide)