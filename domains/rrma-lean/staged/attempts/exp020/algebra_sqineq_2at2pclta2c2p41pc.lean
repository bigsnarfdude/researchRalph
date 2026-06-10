import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

-- RHS - LHS = a² + c² + 4 + 4c - 4a - 2ac = (a - c - 2)² ≥ 0
theorem algebra_sqineq_2at2pclta2c2p41pc (a c : ℝ) :
  2 * a * (2 + c) ≤ a ^ 2 + c ^ 2 + 4 * (1 + c) := by
  nlinarith [sq_nonneg (a - c - 2)]
