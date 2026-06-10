import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_123 (a b : ℕ) (h₀ : 0 < a ∧ 0 < b) (h₁ : a + b = 20) (h₂ : a = 3 * b) :
  a - b = 10 := by
  first
  | solve | linarith [h₀, h₁, h₂]
  | solve | nlinarith [h₀, h₁, h₂]
  | solve | constructor <;> linarith [h₀, h₁, h₂]
  | solve | constructor <;> nlinarith [h₀, h₁, h₂]
  | solve | constructor <;> omega
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | decide
  | solve | native_decide
  | solve | simp only [h₀, h₁, h₂]; ring
  | solve | simp only [h₀, h₁, h₂]; norm_num
  | solve | simp only [h₀, h₁, h₂]; omega
  | solve | simp only [h₀, h₁, h₂]; linarith
  | solve | simp only [h₀, h₁, h₂]; nlinarith
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