import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_327 (a : ℝ) (h₀ : 1 / 5 * abs (9 + 2 * a) < 1) : -7 < a ∧ a < -2 := by
  first
    | constructor <;> linarith [h₀]
    | constructor <;> omega
    | constructor <;> norm_num
    | constructor <;> simp_all
    | simp [abs_of_nonneg, abs_of_nonpos]; norm_num
    | norm_num
    | ring
    | omega
    | linarith
    | simp_all
    | decide