import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem amc12_2001_p2 (a b n : ℕ) (h₀ : 1 ≤ a ∧ a ≤ 9) (h₁ : 0 ≤ b ∧ b ≤ 9) (h₂ : n = 10 * a + b)
  (h₃ : n = a * b + a + b) : b = 9 := by
  have h4 : 9 * a = a * b := by omega
  have ha : a ≠ 0 := by omega
  have := mul_left_cancel₀ ha (show a * 9 = a * b by omega)
  omega
