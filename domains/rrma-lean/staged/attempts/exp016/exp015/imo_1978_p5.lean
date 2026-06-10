import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1978_p5 (n : ℕ) (a : ℕ → ℕ) (h₀ : Function.Injective a) (h₁ : a 0 = 0) (h₂ : 0 < n) :
  (∑ k ∈ Finset.Icc 1 n, (1 : ℝ) / k) ≤ ∑ k ∈ Finset.Icc 1 n, (a k : ℝ) / k ^ 2 := by
  first
  | solve | native_decide
  | solve | decide
  | solve | norm_num
  | solve | simp [h₀, h₁, h₂]; ring
  | solve | simp [h₀, h₁, h₂]; norm_num
  | solve | simp [h₀, h₁, h₂]; native_decide
  | solve | simp; native_decide
  | solve | simp; norm_num
  | solve | simp_all; native_decide
  | solve | omega
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | simp only [h₀, h₁, h₂]; ring
  | solve | simp only [h₀, h₁, h₂]; norm_num
  | solve | simp only [h₀, h₁, h₂]; omega
  | solve | simp only [h₀, h₁, h₂]; linarith
  | solve | simp only [h₀, h₁, h₂]; nlinarith
  | solve | linarith [h₀, h₁, h₂]
  | solve | nlinarith [h₀, h₁, h₂]
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