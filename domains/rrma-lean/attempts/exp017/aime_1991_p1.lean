import Mathlib

set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

theorem aime_1991_p1 (x y : ℕ) (h₀ : 0 < x ∧ 0 < y) (h₁ : x * y + (x + y) = 71)
  (h₂ : x ^ 2 * y + x * y ^ 2 = 880) : x ^ 2 + y ^ 2 = 146 := by
  have hx := h₀.1
  have hy := h₀.2
  have hxb : x ≤ 35 := by nlinarith [Nat.le_mul_of_pos_right x hy]
  have hyb : y ≤ 35 := by nlinarith [Nat.le_mul_of_pos_right y hx]
  interval_cases x <;> interval_cases y <;> omega
