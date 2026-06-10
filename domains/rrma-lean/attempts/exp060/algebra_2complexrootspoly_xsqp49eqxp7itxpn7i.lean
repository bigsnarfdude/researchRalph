import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem algebra_2complexrootspoly_xsqp49eqxp7itxpn7i (x : ℂ) :
    x ^ 2 + 49 = (x + 7 * Complex.I) * (x + -7 * Complex.I) := by
  have h : Complex.I * Complex.I = -1 := Complex.I_mul_I
  have key : (x + 7 * Complex.I) * (x + -7 * Complex.I) = x ^ 2 - 49 * (Complex.I * Complex.I) := by ring
  rw [h] at key; simp only [mul_neg, mul_one, sub_neg_eq_add] at key; exact key.symm
