import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_2complexrootspoly_xsqp49eqxp7itxpn7i (x : ℂ) :
    x ^ 2 + 49 = (x + 7 * Complex.I) * (x + -7 * Complex.I) := by
  first
    | norm_num
    | native_decide
    | ring
    | omega
    | linarith
    | simp_all
    | decide