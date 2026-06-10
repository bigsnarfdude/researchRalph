import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_192 (q e d : ℂ) (h₀ : q = 11 - 5 * Complex.I) (h₁ : e = 11 + 5 * Complex.I)
    (h₂ : d = 2 * Complex.I) : q * e * d = 292 * Complex.I := by
  first
    | subst_vars; ring
    | subst_vars; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide