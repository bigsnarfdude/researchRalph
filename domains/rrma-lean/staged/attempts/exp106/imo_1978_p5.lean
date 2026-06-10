import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1978_p5 (n : ℕ) (a : ℕ → ℕ) (h₀ : Function.Injective a) (h₁ : a 0 = 0) (h₂ : 0 < n) :
  (∑ k ∈ Finset.Icc 1 n, (1 : ℝ) / k) ≤ ∑ k ∈ Finset.Icc 1 n, (a k : ℝ) / k ^ 2 := by
  first
  | solve | native_decide
  | solve | field_simp; linarith [h₀, h₁, h₂]
  | solve | field_simp; nlinarith [h₀, h₁, h₂]
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; norm_num
  | solve | linarith [h₀, h₁, h₂]
  | solve | nlinarith [h₀, h₁, h₂]
  | solve | nlinarith [sq_nonneg _, h₀, h₁, h₂]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀, h₁, h₂]; ring
  | solve | simp only [h₀, h₁, h₂]; norm_num
  | solve | simp only [h₀, h₁, h₂]; omega
  | solve | simp only [h₀, h₁, h₂]; linarith
  | solve | simp only [h₀, h₁, h₂]; nlinarith
  | solve | decide
  | solve | simp; norm_num
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | norm_num
  | solve | ring
  | solve | simp; ring
  | solve | simp; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num