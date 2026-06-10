import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_110 (q e : ℂ) (h₀ : q = 2 - 2 * Complex.I) (h₁ : e = 5 + 5 * Complex.I) :
    q * e = 20 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg q, sq_nonneg e, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (q - e), sq_nonneg (q + e), mul_self_nonneg (q - e)]
    | simp_all [*]
    | decide