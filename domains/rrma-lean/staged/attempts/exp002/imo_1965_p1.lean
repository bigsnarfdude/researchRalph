import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem imo_1965_p1 (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ 2 * π)
  (h₂ : 2 * Real.cos x ≤ abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))))
  (h₃ : abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ≤ Real.sqrt 2) :
  π / 4 ≤ x ∧ x ≤ 7 * π / 4 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg h₂, sq_nonneg (x - h₀), sq_nonneg (x + h₀), mul_self_nonneg (x - h₀)]
    | simp_all
    | decide)