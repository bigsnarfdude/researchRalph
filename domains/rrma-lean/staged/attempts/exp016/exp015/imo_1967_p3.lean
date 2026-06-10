import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1967_p3 (k m n : ℕ) (c : ℕ → ℕ) (h₀ : 0 < k ∧ 0 < m ∧ 0 < n)
  (h₁ : ∀ s, c s = s * (s + 1)) (h₂ : Nat.Prime (k + m + 1)) (h₃ : n + 1 < k + m + 1) :
  (∏ i ∈ Finset.Icc 1 n, c i) ∣ ∏ i ∈ Finset.Icc 1 n, c (m + i) - c k := by
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
  | solve | simp [h₀, h₁, h₂, h₃]; ring
  | solve | simp [h₀, h₁, h₂, h₃]; norm_num
  | solve | simp [h₀, h₁, h₂, h₃]; native_decide
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
  | solve | simp only [h₀, h₁, h₂, h₃]; ring
  | solve | simp only [h₀, h₁, h₂, h₃]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃]; omega
  | solve | simp only [h₀, h₁, h₂, h₃]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃]; nlinarith
  | solve | linarith [h₀, h₁, h₂, h₃]
  | solve | nlinarith [h₀, h₁, h₂, h₃]
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