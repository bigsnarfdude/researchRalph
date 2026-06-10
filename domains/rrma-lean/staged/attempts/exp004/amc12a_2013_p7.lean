import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2013_p7 (s : ℕ → ℝ) (h₀ : ∀ n, s (n + 2) = s (n + 1) + s n) (h₁ : s 9 = 110)
    (h₂ : s 7 = 42) : s 4 = 10 := by
  first
    | omega
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide