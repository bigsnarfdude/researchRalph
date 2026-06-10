import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_10 : abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10 := by
  first
    | norm_num
    | field_simp; ring
    | field_simp; linarith
    | field_simp; norm_num
    | simp [abs_of_nonneg, abs_of_nonpos]; norm_num
    | ring
    | omega
    | linarith
    | simp_all
    | decide