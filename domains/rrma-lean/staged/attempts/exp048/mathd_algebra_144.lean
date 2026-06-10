import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_144 (a b c d : ℕ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h₀ : (c : ℤ) - b = d)
    (h₁ : (b : ℤ) - a = d) (h₂ : a + b + c = 60) (h₃ : a + b > c) : d < 10 := by
  first
  | solve | constructor <;> omega
  | solve | constructor <;> norm_num
  | solve | constructor <;> ring
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]; ring
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]; norm_num
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]; linarith
  | solve | simp only [h₀, h₀, h₁, h₂, h₃]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₀, h₁, h₂, h₃]
  | solve | nlinarith [h₀, h₀, h₁, h₂, h₃]
  | solve | norm_num
  | solve | ring
  | solve | decide
  | solve | simp
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | simp; linarith
  | solve | norm_num; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num