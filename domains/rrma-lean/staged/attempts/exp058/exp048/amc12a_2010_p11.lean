import Mathlib

set_option maxHeartbeats 16000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2010_p11 (x b : ℝ) (h₀ : 0 < b) (h₁ : (7 : ℝ) ^ (x + 7) = 8 ^ x)
  (h₂ : x = Real.logb b (7 ^ 7)) : b = 8 / 7 := by
  have h7pos : (0 : ℝ) < 7 := by norm_num
  have hlog7 : Real.log 7 > 0 := Real.log_pos (by norm_num)
  have hlog8 : Real.log 8 > 0 := Real.log_pos (by norm_num)
  have h_log : (x + 7) * Real.log 7 = x * Real.log 8 := by
    have := congr_arg Real.log h₁
    rw [Real.log_rpow (by norm_num : (0:ℝ) < 7), Real.log_rpow (by norm_num : (0:ℝ) < 8)] at this
    linarith
  have hx : x * (Real.log 8 - Real.log 7) = 7 * Real.log 7 := by linarith
  have hlog87 : Real.log 8 - Real.log 7 > 0 := by
    linarith [Real.log_lt_log (by norm_num : (0:ℝ) < 7) (by norm_num : (7:ℝ) < 8)]
  have hx_pos : 0 < x := by
    by_contra h; push_neg at h; nlinarith [hlog87]
  -- b > 1 (from x > 0 and x = log(7^7)/log(b), log(7^7) > 0)
  have hb_gt1 : 1 < b := by
    by_contra h; push_neg at h
    have hlogb_le : Real.log b ≤ 0 := Real.log_nonpos (le_of_lt h₀) h
    have hlog77_pos : 0 < Real.log (7 ^ 7 : ℝ) := by rw [Real.log_pow]; push_cast; nlinarith
    have : x ≤ 0 := by
      rw [h₂, Real.logb]; exact div_nonpos_of_nonneg_of_nonpos (le_of_lt hlog77_pos) hlogb_le
    linarith
  have hlogb_pos : Real.log b > 0 := Real.log_pos hb_gt1
  -- x * log(b) = 7 * log 7
  have h_xlogb : x * Real.log b = 7 * Real.log 7 := by
    rw [h₂, Real.logb, div_mul_cancel₀ _ (ne_of_gt hlogb_pos), Real.log_pow]; push_cast; ring
  -- log(b) = log(8) - log(7) = log(8/7)
  have hlogb : Real.log b = Real.log 8 - Real.log 7 := by
    have := mul_left_cancel₀ (ne_of_gt hx_pos) (show x * Real.log b = x * (Real.log 8 - Real.log 7) from by linarith)
    linarith
  rw [← Real.log_div (by norm_num : (8:ℝ) ≠ 0) (by norm_num : (7:ℝ) ≠ 0)] at hlogb
  exact Real.log_injOn_pos (Set.mem_Ioi.mpr h₀) (Set.mem_Ioi.mpr (by norm_num : (0:ℝ) < 8/7)) hlogb
