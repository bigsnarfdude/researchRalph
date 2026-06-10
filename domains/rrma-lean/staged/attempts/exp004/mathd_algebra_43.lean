import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_43 (a b : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = a * x + b) (h₁ : f 7 = 4)
  (h₂ : f 6 = 3) : f 3 = 0 := by
  first
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide