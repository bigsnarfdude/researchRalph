import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1977_p5 (a b q r : ℕ) (h₀ : r < a + b) (h₁ : a ^ 2 + b ^ 2 = (a + b) * q + r)
  (h₂ : q ^ 2 + r = 1977) :
  abs ((a : ℤ) - 22) = 15 ∧ abs ((b : ℤ) - 22) = 28 ∨
    abs ((a : ℤ) - 22) = 28 ∧ abs ((b : ℤ) - 22) = 15 := by
  first
    | omega
    | constructor <;> linarith [h₀, h₁, h₂]
    | constructor <;> omega
    | constructor <;> norm_num
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide