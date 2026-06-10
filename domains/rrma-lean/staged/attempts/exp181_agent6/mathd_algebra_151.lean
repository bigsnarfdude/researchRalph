import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_151 : Int.ceil (Real.sqrt 27) - Int.floor (Real.sqrt 26) = 1 := by
  have hsqrt27_lo : 5 < Real.sqrt 27 := by
    rw [show (5 : ℝ) = Real.sqrt (5^2) from by rw [Real.sqrt_sq (by norm_num : (5:ℝ) ≥ 0)]]
    exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
  have hsqrt27_hi : Real.sqrt 27 ≤ 6 := by
    rw [show (6 : ℝ) = Real.sqrt (6^2) from by rw [Real.sqrt_sq (by norm_num : (6:ℝ) ≥ 0)]]
    exact Real.sqrt_le_sqrt (by norm_num)
  have hsqrt26_lo : 5 ≤ Real.sqrt 26 := by
    rw [show (5 : ℝ) = Real.sqrt (5^2) from by rw [Real.sqrt_sq (by norm_num : (5:ℝ) ≥ 0)]]
    exact Real.sqrt_le_sqrt (by norm_num)
  have hsqrt26_hi : Real.sqrt 26 < 6 := by
    rw [show (6 : ℝ) = Real.sqrt (6^2) from by rw [Real.sqrt_sq (by norm_num : (6:ℝ) ≥ 0)]]
    exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
  have hceil : Int.ceil (Real.sqrt 27) = 6 := by
    rw [Int.ceil_eq_iff]
    · exact ⟨by push_cast; linarith, by push_cast; linarith⟩
  have hfloor : Int.floor (Real.sqrt 26) = 5 := by
    rw [Int.floor_eq_iff]
    · exact ⟨by push_cast; linarith, by push_cast; linarith⟩
  simp [hceil, hfloor]
