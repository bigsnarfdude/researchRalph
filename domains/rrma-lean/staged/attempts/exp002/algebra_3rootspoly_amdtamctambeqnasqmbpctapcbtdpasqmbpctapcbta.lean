import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem algebra_3rootspoly_amdtamctambeqnasqmbpctapcbtdpasqmbpctapcbta (b c d a : ℂ) :
    (a - d) * (a - c) * (a - b) =
      -((a ^ 2 - (b + c) * a + c * b) * d) + (a ^ 2 - (b + c) * a + c * b) * a := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg b, sq_nonneg c, sq_nonneg d, sq_nonneg a, sq_nonneg (b - c), sq_nonneg (b + c), mul_self_nonneg (b - c)]
    | simp_all [*]
    | decide