import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12b_2003_p17 (x y : ℝ) (h₀ : 0 < x ∧ 0 < y) (h₁ : Real.log (x * y ^ 3) = 1)
  (h₂ : Real.log (x ^ 2 * y) = 1) : Real.log (x * y) = 3 / 5 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg y, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (x - y), sq_nonneg (x + y), mul_self_nonneg (x - y)]
    | simp_all
    | decide)