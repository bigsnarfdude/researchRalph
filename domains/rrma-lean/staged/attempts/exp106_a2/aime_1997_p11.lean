import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aime_1997_p11 (x : ℝ)
    (h₀ :
      x =
        (∑ n ∈ Finset.Icc (1 : ℕ) 44, Real.cos (n * π / 180)) /
          ∑ n ∈ Finset.Icc (1 : ℕ) 44, Real.sin (n * π / 180)) :
    Int.floor (100 * x) = 241 := by
  first
  | solve | native_decide
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
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
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp; ring
  | solve | simp; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num