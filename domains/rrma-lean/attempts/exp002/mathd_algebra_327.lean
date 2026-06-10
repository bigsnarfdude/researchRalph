import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_327 (a : ℝ) (h₀ : 1 / 5 * abs (9 + 2 * a) < 1) : -7 < a ∧ a < -2 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg h₀, sq_nonneg (a - h₀), sq_nonneg (a + h₀), mul_self_nonneg (a - h₀)]
    | simp_all
    | decide)