import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_192 (q e d : ℂ) (h₀ : q = 11 - 5 * Complex.I) (h₁ : e = 11 + 5 * Complex.I)
    (h₂ : d = 2 * Complex.I) : q * e * d = 292 * Complex.I := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg q, sq_nonneg e, sq_nonneg d, sq_nonneg h₀, sq_nonneg (q - e), sq_nonneg (q + e), mul_self_nonneg (q - e)]
    | simp_all [*]
    | decide