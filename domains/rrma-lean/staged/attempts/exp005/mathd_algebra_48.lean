import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_48 (q e : ℂ) (h₀ : q = 9 - 4 * Complex.I) (h₁ : e = -3 - 4 * Complex.I) :
  q - e = 12 := by
  first
    | subst_vars; norm_num [Complex.ext_iff]
    | subst_vars; ring
    | subst_vars; apply Complex.ext <;> simp <;> ring
    | ring
    | norm_num
    | simp_all