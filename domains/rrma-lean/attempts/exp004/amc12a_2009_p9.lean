import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2009_p9 (a b c : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f (x + 3) = 3 * x ^ 2 + 7 * x + 4)
  (h₁ : ∀ x, f x = a * x ^ 2 + b * x + c) : a + b + c = 2 := by
  first
    | simp only [h₀, h₁]; ring
    | simp only [h₀, h₁]; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide