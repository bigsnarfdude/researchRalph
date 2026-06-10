import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_22 : Real.logb (5 ^ 2) (5 ^ 4) = 2 := by
  rw [Real.logb, Real.log_pow, Real.log_pow]
  have h5 : Real.log 5 ≠ 0 := Real.log_ne_zero_of_pos_of_ne_one (by norm_num) (by norm_num)
  field_simp
  ring
