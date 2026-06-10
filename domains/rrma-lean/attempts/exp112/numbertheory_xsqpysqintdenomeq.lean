import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat

theorem numbertheory_xsqpysqintdenomeq (x y : ℚ) (h₀ : (x ^ 2 + y ^ 2).den = 1) : x.den = y.den := by
  set s := x ^ 2 + y ^ 2
  have hs_eq : (s.num : ℚ) = s := (Rat.den_eq_one_iff s).mp h₀
  have hden : (y ^ 2).den = (x ^ 2).den := by
    have : y ^ 2 = -x ^ 2 + ↑s.num := by rw [hs_eq]; ring
    rw [this]; exact Rat.add_intCast_den _ _
  rw [Rat.den_pow, Rat.den_pow] at hden
  exact Nat.pow_left_injective two_ne_zero hden.symm
