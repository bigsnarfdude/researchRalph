import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem algebra_xmysqpymzsqpzmxsqeqxyz_xpypzp6dvdx3y3z3 (x y z : ℤ)
  (h₀ : (x - y) ^ 2 + (y - z) ^ 2 + (z - x) ^ 2 = x * y * z) :
  x + y + z + 6 ∣ x ^ 3 + y ^ 3 + z ^ 3 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg y, sq_nonneg z, sq_nonneg h₀, sq_nonneg (x - y), sq_nonneg (x + y), mul_self_nonneg (x - y)]
    | simp_all [*]
    | decide