import Mathlib

set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

theorem algebra_amgm_sqrtxymulxmyeqxpy_xpygeq4 (x y : ℝ) (h₀ : 0 < x ∧ 0 < y) (h₁ : y ≤ x)
  (h₂ : Real.sqrt (x * y) * (x - y) = x + y) : x + y ≥ 4 := by
  have hx := h₀.1
  have hy := h₀.2
  have hxy_pos : x * y > 0 := mul_pos hx hy
  have hsqrt_sq : Real.sqrt (x * y) ^ 2 = x * y := Real.sq_sqrt (le_of_lt hxy_pos)
  have hsqrt_pos : Real.sqrt (x * y) > 0 := Real.sqrt_pos.mpr hxy_pos
  -- Square h₂: xy * (x-y)² = (x+y)²
  have hsq : x * y * (x - y) ^ 2 = (x + y) ^ 2 := by
    have := congr_arg (· ^ 2) h₂
    simp only [mul_pow] at this
    rw [hsqrt_sq] at this
    linarith
  -- From xy(x-y)² = (x+y)² and x+y > 0:
  -- (x+y)² ≥ 16 follows from (xy-2)² ≥ 0 and xy > 1
  -- First show xy > 1
  have hxy_gt1 : x * y > 1 := by
    by_contra h
    push_neg at h
    -- If xy ≤ 1 and (x-y) ≥ 0: xy(x-y)² ≤ (x-y)²
    -- Also (x+y)² ≥ (x-y)² + 4xy (expand both)
    -- So (x+y)² > 0 = xy(x-y)² when xy=0, contradiction
    -- More carefully: if xy ≤ 1, then xy(x-y)² ≤ (x-y)²
    -- But (x+y)² = (x-y)² + 4xy, so (x+y)² = xy(x-y)² gives
    -- (x-y)² + 4xy = xy(x-y)², so (x-y)²(xy-1) = 4xy > 0
    -- Since xy > 0, we need xy > 1. Contradiction with h.
    have key : (x - y) ^ 2 * (x * y - 1) = 4 * (x * y) := by nlinarith [hsq]
    have : (x - y) ^ 2 * (x * y - 1) ≤ 0 := by
      apply mul_nonpos_of_nonneg_of_nonpos (sq_nonneg _)
      linarith
    linarith
  -- Now: (x+y)² = xy(x-y)² = xy((x+y)² - 4xy)
  -- Let s = x+y, p = xy: s² = p(s²-4p) = ps² - 4p²
  -- s²(p-1) = 4p²
  -- s² = 4p²/(p-1)
  -- s² - 16 = 4p²/(p-1) - 16 = (4p² - 16p + 16)/(p-1) = 4(p-2)²/(p-1) ≥ 0
  have key : (x + y) ^ 2 * (x * y - 1) = 4 * (x * y) ^ 2 := by nlinarith [hsq]
  -- 4(xy)² ≥ 16(xy-1) from (xy-2)² ≥ 0
  have step : 4 * (x * y) ^ 2 ≥ 16 * (x * y - 1) := by nlinarith [sq_nonneg (x * y - 2)]
  -- So (x+y)²(xy-1) ≥ 16(xy-1), and xy-1 > 0 → (x+y)² ≥ 16
  have hxy1_pos : x * y - 1 > 0 := by linarith
  have h16 : 16 * (x * y - 1) ≤ (x + y) ^ 2 * (x * y - 1) := by linarith [key, step]
  have hsq_ge : (16 : ℝ) ≤ (x + y) ^ 2 := (mul_le_mul_right hxy1_pos).mp h16
  have hsum_pos : x + y > 0 := by linarith
  nlinarith [sq_nonneg (x + y - 4)]
