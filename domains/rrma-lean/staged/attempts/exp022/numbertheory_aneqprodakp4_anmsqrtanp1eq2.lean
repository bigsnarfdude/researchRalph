import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem numbertheory_aneqprodakp4_anmsqrtanp1eq2 (a : ℕ → ℝ) (h₀ : a 0 = 1)
  (h₁ : ∀ n, a (n + 1) = (∏ k ∈ Finset.range (n + 1), a k) + 4) :
  ∀ n ≥ 1, a n - Real.sqrt (a (n + 1)) = 2 := by
  first
  | solve | native_decide
  | solve | decide
  | solve | norm_num
  | solve | simp only [h₁]
  | solve | simp only [h₁]; ring
  | solve | simp only [h₁]; norm_num
  | solve | simp [h₁]; ring
  | solve | simp [h₁]; norm_num
  | solve | simp only [h₁]; simp [Finset.sum_add_adjacent]; ring
  | solve | simp only [h₁]; rw [Finset.sum_sub_distrib]; simp; ring
  | solve | simp [h₀, h₁]; ring
  | solve | simp [h₀, h₁]; norm_num
  | solve | simp [h₀, h₁]; native_decide
  | solve | simp [Finset.prod_Icc_succ]
  | solve | simp only [h₁]; simp [Finset.prod_div_distrib]; norm_num
  | solve | simp; native_decide
  | solve | simp; norm_num
  | solve | simp_all; native_decide
  | solve | omega
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | simp only [h₀, h₁]; ring
  | solve | simp only [h₀, h₁]; norm_num
  | solve | simp only [h₀, h₁]; omega
  | solve | simp only [h₀, h₁]; linarith
  | solve | simp only [h₀, h₁]; nlinarith
  | solve | linarith [h₀, h₁]
  | solve | nlinarith [h₀, h₁]
  | solve | push_cast; ring
  | solve | push_cast; norm_num
  | solve | push_cast; omega
  | solve | field_simp; ring
  | solve | field_simp; norm_num
  | solve | ring_nf; norm_num
  | solve | ring_nf; omega
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; norm_num
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith