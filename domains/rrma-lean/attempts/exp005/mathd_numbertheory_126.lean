import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_126 (x a : ℕ) (h₀ : 0 < x ∧ 0 < a) (h₁ : Nat.gcd a 40 = x + 3)
  (h₂ : Nat.lcm a 40 = x * (x + 3))
  (h₃ : ∀ b : ℕ, 0 < b → Nat.gcd b 40 = x + 3 ∧ Nat.lcm b 40 = x * (x + 3) → a ≤ b) : a = 8 := by
  first
    | omega
    | norm_num
    | native_decide
    | decide
    | simp only [h₃] at *; nlinarith
    | simp only [h₃] at *; linarith
    | simp only [h₃] at *; omega
    | simp only [h₃] at *; norm_num
    | simp only [h₃]; ring
    | simp only [h₃]; norm_num
    | ring
    | linarith