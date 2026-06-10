import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2010_p22 (x : ℝ) : 49 ≤ ∑ k ∈ (Finset.Icc (1:ℤ) (119:ℤ)), abs (k * x - 1) := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x]
    | simp_all [*]
    | decide