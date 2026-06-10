import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_140 (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : ∀ x, 24 * x ^ 2 - 19 * x - 35 = (a * x - 5) * (2 * (b * x) + c)) : a * b - 3 * c = -9 := by
  first
    | simp only [h₁]; ring
    | simp only [h₁]; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide