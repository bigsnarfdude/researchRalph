import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) :
  ((∑ k ∈ Finset.Icc 1 12, z ^ k ^ 2) * (∑ k ∈ Finset.Icc 1 12, 1 / z ^ k ^ 2)) = 36 := by
  first
    | subst_vars; ring
    | subst_vars; norm_num
    | field_simp; linarith [h₀]
    | field_simp; ring
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide