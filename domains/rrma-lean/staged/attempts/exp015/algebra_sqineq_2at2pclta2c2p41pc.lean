import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_sqineq_2at2pclta2c2p41pc (a c : ℝ) :
  2 * a * (2 + c) ≤ a ^ 2 + c ^ 2 + 4 * (1 + c) := by
  -- 2a(2+c) ≤ a² + c² + 4 + 4c
  -- 0 ≤ a² - 2a(2+c) + c² + 4 + 4c
  -- 0 ≤ a² - 4a - 2ac + c² + 4c + 4
  -- 0 ≤ (a-2)² + (c-a)² + ... let me try different witnesses
  nlinarith [sq_nonneg (a - c - 2), sq_nonneg (a - 2), sq_nonneg c, sq_nonneg (c - a)]
