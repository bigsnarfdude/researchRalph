import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_668 (l r : ZMod 7) (h₀ : l = (2 + 3)⁻¹) (h₁ : r = 2⁻¹ + 3⁻¹) :
  l - r = 1 := by
  first
    | simp only [h₀, h₁]; ring
    | simp only [h₀, h₁]; norm_num
    | simp only [h₀, h₁]; omega
    | simp only [h₀, h₁]; linarith
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide