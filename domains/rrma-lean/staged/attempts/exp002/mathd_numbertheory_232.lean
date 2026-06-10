import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_232 (x y z : ZMod 31) (h₀ : x = 3⁻¹) (h₁ : y = 5⁻¹)
  (h₂ : z = (x + y)⁻¹) : z = 29 := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg y, sq_nonneg z, sq_nonneg h₀, sq_nonneg (x - y), sq_nonneg (x + y), mul_self_nonneg (x - y)]
    | simp_all [*]
    | decide