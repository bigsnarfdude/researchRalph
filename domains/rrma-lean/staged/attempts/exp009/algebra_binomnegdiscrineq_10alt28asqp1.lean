import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

-- 28a² - 10a + 1 ≥ 0. Discriminant = 100 - 112 = -12 < 0, so always true.
-- Completing square: 28a² - 10a + 1 = 28(a - 5/28)² + 1 - 25/28 = 28(a-5/28)² + 3/28
-- Alternatively: 28a²-10a+1 = 4(2a-1)² + 12a² - 2a - 3 ... hmm
-- Let's try: 28a²-10a+1 = (4a-1)² + 12a² - 2a = (4a-1)² + 2a(6a-1)
-- Or: 28a²-10a+1 ≥ 0 ⟺ (5a-1)² ≥ 1-3a² which follows from 3a²≥0 and extra
theorem algebra_binomnegdiscrineq_10alt28asqp1 (a : ℝ) : 10 * a ≤ 28 * a ^ 2 + 1 := by
  nlinarith [sq_nonneg (4 * a - 1), sq_nonneg a, sq_nonneg (2 * a)]
