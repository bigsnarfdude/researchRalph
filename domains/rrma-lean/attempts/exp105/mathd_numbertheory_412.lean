import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_412 (x y : ℤ) (h₀ : x % 19 = 4) (h₁ : y % 19 = 7) :
  (x + 1) ^ 2 * (y + 5) ^ 3 % 19 = 13 := by
  have hx : ∃ a, x = 4 + 19 * a := ⟨(x - 4) / 19, by omega⟩
  have hy : ∃ b, y = 7 + 19 * b := ⟨(y - 7) / 19, by omega⟩
  obtain ⟨a, rfl⟩ := hx
  obtain ⟨b, rfl⟩ := hy
  ring_nf
  omega
