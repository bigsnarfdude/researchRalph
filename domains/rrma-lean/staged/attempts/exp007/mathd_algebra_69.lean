import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_69 (rows seats : ℕ) (h₀ : rows * seats = 450)
  (h₁ : (rows + 5) * (seats - 3) = 450) : rows = 25 := by
  have h2 : 5 * seats = 3 * rows + 15 := by omega
  have h3 : rows * (3 * rows + 15) = 2250 := by omega
  nlinarith [sq_nonneg (rows - 25)]
