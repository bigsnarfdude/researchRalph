import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12_2000_p5 (x p : ℝ) (h₀ : x < 2) (h₁ : abs (x - 2) = p) : x - p = 2 - 2 * p := by
  first
    | simp [abs_of_nonneg, abs_of_nonpos]; norm_num
    | norm_num
    | ring
    | omega
    | linarith
    | simp_all
    | decide