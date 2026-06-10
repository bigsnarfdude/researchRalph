import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12b_2002_p11 (a b : ℕ) (h₀ : Nat.Prime a) (h₁ : Nat.Prime b) (h₂ : Nat.Prime (a + b))
  (h₃ : Nat.Prime (a - b)) : Nat.Prime (a + b + (a - b + (a + b))) := by
  first
    | omega
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide