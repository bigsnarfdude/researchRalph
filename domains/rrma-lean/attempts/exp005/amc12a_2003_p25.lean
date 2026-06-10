import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2003_p25 (a b : ℝ) (f : ℝ → ℝ) (h₀ : 0 < b)
  (h₁ : ∀ x, f x = Real.sqrt (a * x ^ 2 + b * x)) (h₂ : { x | 0 ≤ f x } = f '' { x | 0 ≤ f x }) :
  a = 0 ∨ a = -4 := by
  first
    | simp only [h₁] at *; nlinarith
    | simp only [h₁] at *; linarith
    | simp only [h₁] at *; omega
    | simp only [h₁] at *; norm_num
    | simp only [h₁]; ring
    | simp only [h₁]; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide