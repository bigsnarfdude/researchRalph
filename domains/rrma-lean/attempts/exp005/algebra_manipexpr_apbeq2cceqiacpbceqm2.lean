import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_manipexpr_apbeq2cceqiacpbceqm2 (a b c : ℂ) (h₀ : a + b = 2 * c)
  (h₁ : c = Complex.I) : a * c + b * c = -2 := by
  first
    | subst_vars; norm_num [Complex.ext_iff]
    | subst_vars; ring
    | subst_vars; apply Complex.ext <;> simp <;> ring
    | ring
    | norm_num
    | simp_all