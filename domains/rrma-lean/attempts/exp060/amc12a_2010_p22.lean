import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
-- Currently unsolved (∑|kx-1| ≥ 49)
theorem amc12a_2010_p22 (x : ℝ) : 49 ≤ ∑ k ∈ (Finset.Icc (1:ℤ) (119:ℤ)), abs (k * x - 1) := by
  omega
