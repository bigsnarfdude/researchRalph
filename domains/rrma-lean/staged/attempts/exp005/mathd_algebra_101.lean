import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_101 (x : ℝ) (h₀ : x ^ 2 - 5 * x - 4 ≤ 10) : x ≥ -2 ∧ x ≤ 7 := by
  first
    | constructor <;> linarith [h₀]
    | constructor <;> omega
    | constructor <;> norm_num
    | constructor <;> simp_all
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide