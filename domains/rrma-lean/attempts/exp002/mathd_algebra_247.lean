import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_247 (t s : ℝ) (n : ℤ) (h₀ : t = 2 * s - s ^ 2) (h₁ : s = n ^ 2 - 2 ^ n + 1)
  (_ : n = 3) : t = 0 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg t, sq_nonneg s, sq_nonneg n, sq_nonneg h₀, sq_nonneg (t - s), sq_nonneg (t + s), mul_self_nonneg (t - s)]
    | simp_all [*]
    | decide