import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_437 (x y : ℝ) (n : ℤ) (h₀ : x ^ 3 = -45) (h₁ : y ^ 3 = -101) (h₂ : x < n)
  (h₃ : ↑n < y) : n = -4 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg y, sq_nonneg n, sq_nonneg h₀, sq_nonneg (x - y), sq_nonneg (x + y), mul_self_nonneg (x - y)]
    | simp_all [*]
    | decide