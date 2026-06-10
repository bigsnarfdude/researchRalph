import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat
theorem amc12a_2016_p2 (x : ℝ) (h₀ : (10 : ℝ) ^ x * 100 ^ (2 * x) = 1000 ^ 5) : x = 3 := by
  have h10 : (0 : ℝ) < 10 := by norm_num
  have h100 : (0 : ℝ) < 100 := by norm_num
  have hlog := congr_arg Real.log h₀
  rw [Real.log_mul (ne_of_gt (rpow_pos_of_pos h10 x)) (ne_of_gt (rpow_pos_of_pos h100 (2*x)))] at hlog
  rw [Real.log_rpow h10, Real.log_rpow h100] at hlog
  rw [show (1000 : ℝ) = 10 ^ (3 : ℕ) from by norm_num, ← pow_mul, Real.log_pow] at hlog
  have hlog100 : Real.log 100 = 2 * Real.log 10 := by
    rw [show (100 : ℝ) = 10 ^ 2 from by norm_num, Real.log_pow]; ring
  rw [hlog100] at hlog
  have hln10 : Real.log 10 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  have key : (5 * x - 15) * Real.log 10 = 0 := by push_cast at hlog; nlinarith
  rcases mul_eq_zero.mp key with h | h
  · linarith
  · exact absurd h hln10
