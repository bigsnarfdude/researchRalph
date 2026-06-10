import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_sqineq_2at2pclta2c2p41pc (a c : ℝ) :
  2 * a * (2 + c) ≤ a ^ 2 + c ^ 2 + 4 * (1 + c) := by
  first
    | norm_num
    | ring
    | omega
    | linarith
    | simp_all
    | decide