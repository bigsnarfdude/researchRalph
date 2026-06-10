import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_144 (a b c d : ℕ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h₀ : (c : ℤ) - b = d)
    (h₁ : (b : ℤ) - a = d) (h₂ : a + b + c = 60) (h₃ : a + b > c) : d < 10 := by
  first
  | solve | linarith [h₀, h₀, h₁, h₂, h₃]
  | solve | nlinarith [h₀, h₀, h₁, h₂, h₃]
  | solve | constructor <;> linarith [h₀, h₀, h₁, h₂, h₃]
  | solve | constructor <;> nlinarith [h₀, h₀, h₁, h₂, h₃]
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
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]; ring
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]; norm_num
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]; omega
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]; linarith
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]; nlinarith
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