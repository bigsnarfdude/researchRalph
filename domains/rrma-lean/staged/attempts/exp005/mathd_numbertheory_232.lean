import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_232 (x y z : ZMod 31) (h₀ : x = 3⁻¹) (h₁ : y = 5⁻¹)
  (h₂ : z = (x + y)⁻¹) : z = 29 := by
  first
    | simp only [h₀, h₁, h₂]; ring
    | simp only [h₀, h₁, h₂]; norm_num
    | simp only [h₀, h₁, h₂]; omega
    | simp only [h₀, h₁, h₂]; linarith
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide