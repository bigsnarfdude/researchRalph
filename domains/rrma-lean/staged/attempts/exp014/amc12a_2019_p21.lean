import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) :
  ((∑ k ∈ Finset.Icc 1 12, z ^ k ^ 2) * (∑ k ∈ Finset.Icc 1 12, 1 / z ^ k ^ 2)) = 36 := by
  first
  | solve | native_decide
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num [Complex.ext_iff, Complex.I_sq]
  | solve | subst_vars; simp [Complex.ext_iff, Complex.I_sq]; ring
  | solve | ring
  | solve | norm_num
  | solve | simp only [h₀]; ring
  | solve | field_simp; linarith [h₀]
  | solve | field_simp; nlinarith [h₀]
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; norm_num
  | solve | decide
  | solve | simp; norm_num
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | omega
  | solve | linarith
  | solve | nlinarith
  | solve | simp; ring
  | solve | simp; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num