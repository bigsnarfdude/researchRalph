import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12b_2002_p6 (a b : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0)
  (h₁ : ∀ x, x ^ 2 + a * x + b = (x - a) * (x - b)) : a = 1 ∧ b = -2 := by
  first
    | simp only [h₁]; ring
    | simp only [h₁]; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide