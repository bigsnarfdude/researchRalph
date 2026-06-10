import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

-- Goal: 2a(2+c) ≤ a² + c² + 4(1+c)
-- Equiv: 0 ≤ a² - 2a(2+c) + c² + 4 + 4c
-- = a² - 4a - 2ac + c² + 4 + 4c
-- = (a-2)² - 2ac + c² + 4c = (a-2)² + c² - 2ac + 4c
-- = (a-2)² + (c-a)² - a² + 4c + 4a - 4 ... getting complicated
-- Try: a² + c² + 4 + 4c - 4a - 2ac = (a-c-2)² + 2c(c+2) - (c+2)² + ...
-- Simpler: try (a - c - 2)² = a² + c² + 4 - 2ac - 4a + 4c. That's exactly it!
theorem algebra_sqineq_2at2pclta2c2p41pc (a c : ℝ) :
  2 * a * (2 + c) ≤ a ^ 2 + c ^ 2 + 4 * (1 + c) := by
  nlinarith [sq_nonneg (a - c - 2)]
