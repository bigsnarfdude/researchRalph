import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem algebra_2varlineareq_xpeeq7_2xpeeq3_eeq11_xeqn4 (x e : ℂ) (h₀ : x + e = 7)
  (h₁ : 2 * x + e = 3) : e = 11 ∧ x = -4 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg e, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (x - e), sq_nonneg (x + e), mul_self_nonneg (x - e)]
    | simp_all
    | decide)