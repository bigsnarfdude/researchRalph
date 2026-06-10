import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem aime_1983_p9 (x : ℝ) (h₀ : 0 < x ∧ x < Real.pi) :
  12 ≤ (9 * (x ^ 2 * Real.sin x ^ 2) + 4) / (x * Real.sin x) := by
  have ht : 0 < x * Real.sin x :=
    mul_pos h₀.1 (Real.sin_pos_of_pos_of_lt_pi h₀.1 h₀.2)
  rw [le_div_iff₀ ht]
  nlinarith [sq_nonneg (3 * x * Real.sin x - 2)]
