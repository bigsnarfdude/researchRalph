import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2016_p3 (f : ℝ → ℝ → ℝ)
  (h₀ : ∀ x, ∀ (y) (_ : y ≠ 0), f x y = x - y * Int.floor (x / y)) :
  f (3 / 8) (-(2 / 5)) = -(1 / 40) := by
  first
  | solve | simp only [h₀] at *; ring
  | solve | simp only [h₀] at *; norm_num
  | solve | simp only [h₀] at *; omega
  | solve | simp only [h₀] at *; linarith
  | solve | simp only [h₀] at *; nlinarith
  | solve | simp only [h₀] at *; field_simp; ring
  | solve | simp only [h₀] at *; field_simp; linarith
  | solve | linear_combination h₀
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
  | solve | decide
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num