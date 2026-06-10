import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2010_p22 (x : ℝ) : 49 ≤ ∑ k ∈ (Finset.Icc (1:ℤ) (119:ℤ)), abs (k * x - 1) := by
  first
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | norm_num
    | simp [abs_of_nonneg, abs_of_nonpos]; norm_num
    | ring
    | omega
    | linarith
    | simp_all