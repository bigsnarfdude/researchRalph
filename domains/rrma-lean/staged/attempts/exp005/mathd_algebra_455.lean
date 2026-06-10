import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_455 (x : ℝ) (h₀ : 2 * (2 * (2 * (2 * x))) = 48) : x = 3 := by
  first
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide