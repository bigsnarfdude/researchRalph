import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aimeII_2020_p6 (t : ℕ → ℚ) (h₀ : t 1 = 20) (h₁ : t 2 = 21)
  (h₂ : ∀ n ≥ 3, t n = (5 * t (n - 1) + 1) / (25 * t (n - 2))) :
  ↑(t 2020).den + (t 2020).num = 626 := by
  first
  | solve | simp only [h₂] at *; ring
  | solve | simp only [h₂] at *; norm_num
  | solve | simp only [h₂] at *; omega
  | solve | simp only [h₂] at *; linarith
  | solve | simp only [h₂] at *; nlinarith
  | solve | simp only [h₂] at *; field_simp; ring
  | solve | simp only [h₂] at *; field_simp; linarith
  | solve | linarith [h₀, h₁, h₂]
  | solve | nlinarith [h₀, h₁, h₂]
  | solve | linarith
  | solve | nlinarith
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | decide
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num