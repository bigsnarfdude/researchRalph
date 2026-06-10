import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_668 (l r : ZMod 7) (h₀ : l = (2 + 3)⁻¹) (h₁ : r = 2⁻¹ + 3⁻¹) :
  l - r = 1 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg l, sq_nonneg r, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (l - r), sq_nonneg (l + r), mul_self_nonneg (l - r)]
    | simp_all [*]
    | decide