import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_151 : Int.ceil (Real.sqrt 27) - Int.floor (Real.sqrt 26) = 1 := by
  have h1 : Real.sqrt 27 > 5 := by
    rw [show (5 : ℝ) = Real.sqrt 25 from by rw [Real.sqrt_eq_iff_sq_eq] <;> norm_num]
    exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
  have h2 : Real.sqrt 27 < 6 := by
    rw [show (6 : ℝ) = Real.sqrt 36 from by rw [Real.sqrt_eq_iff_sq_eq] <;> norm_num]
    exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
  have h3 : Real.sqrt 26 > 5 := by
    rw [show (5 : ℝ) = Real.sqrt 25 from by rw [Real.sqrt_eq_iff_sq_eq] <;> norm_num]
    exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
  have h4 : Real.sqrt 26 < 6 := by
    rw [show (6 : ℝ) = Real.sqrt 36 from by rw [Real.sqrt_eq_iff_sq_eq] <;> norm_num]
    exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
  rw [Int.ceil_eq_iff (by linarith)]
  constructor
  · linarith
  · exact_mod_cast h2
  sorry
