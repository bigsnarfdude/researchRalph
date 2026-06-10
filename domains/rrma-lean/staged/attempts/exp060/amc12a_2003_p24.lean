import Mathlib
set_option maxHeartbeats 16000000

open Real

-- IsGreatest {y | ∃ a b, 1 < b ∧ b ≤ a ∧ y = logb a (a/b) + logb b (b/a)} 0
-- logb a (a/b) = 1 - logb a b = 1 - 1/t where t = log a / log b ≥ 1
-- logb b (b/a) = 1 - logb b a = 1 - t
-- y = 2 - t - 1/t ≤ 0 by AM-GM (t+1/t ≥ 2 for t ≥ 1)
-- Equality at t=1, i.e., a=b

theorem amc12a_2003_p24 :
  IsGreatest { y : ℝ | ∃ a b : ℝ, 1 < b ∧ b ≤ a ∧ y = Real.logb a (a / b) + Real.logb b (b / a) }
    0 := by
  constructor
  · -- 0 ∈ S: witness a = b = 2
    refine ⟨2, 2, by norm_num, le_refl _, ?_⟩
    simp [Real.logb, Real.log_div, div_self (ne_of_gt (Real.log_pos (by norm_num : (1:ℝ) < 2)))]
  · -- ∀ y ∈ S, y ≤ 0
    intro y ⟨a, b, hb1, hab, hy⟩
    rw [hy]
    -- logb a (a/b) + logb b (b/a) ≤ 0
    have hb_pos : 0 < b := by linarith
    have ha_pos : 0 < a := by linarith
    have ha1 : 1 < a := by linarith
    have hlog_b : 0 < Real.log b := Real.log_pos hb1
    have hlog_a : 0 < Real.log a := Real.log_pos ha1
    -- logb a (a/b) = log(a/b)/log(a) = (log a - log b)/log a = 1 - log b / log a
    have h1 : Real.logb a (a / b) = 1 - Real.log b / Real.log a := by
      simp [Real.logb, Real.log_div (ne_of_gt ha_pos) (ne_of_gt hb_pos)]
      field_simp
    -- logb b (b/a) = log(b/a)/log(b) = (log b - log a)/log b = 1 - log a / log b
    have h2 : Real.logb b (b / a) = 1 - Real.log a / Real.log b := by
      simp [Real.logb, Real.log_div (ne_of_gt hb_pos) (ne_of_gt ha_pos)]
      field_simp
    rw [h1, h2]
    -- Goal: (1 - log b / log a) + (1 - log a / log b) ≤ 0
    -- i.e., 2 - (log b / log a + log a / log b) ≤ 0
    -- i.e., log b / log a + log a / log b ≥ 2
    -- Let t = log a / log b ≥ 1. Then t + 1/t ≥ 2 by (t-1)² ≥ 0.
    suffices h : Real.log b / Real.log a + Real.log a / Real.log b ≥ 2 by linarith
    have := div_add_div_same (Real.log b) (Real.log a) (Real.log a * Real.log b)
    -- Use AM-GM: x/y + y/x ≥ 2 for x,y > 0
    have key : 0 < Real.log a * Real.log b := mul_pos hlog_a hlog_b
    rw [ge_iff_le, ← sub_nonneg]
    have : Real.log b / Real.log a + Real.log a / Real.log b - 2 =
      (Real.log a - Real.log b) ^ 2 / (Real.log a * Real.log b) := by field_simp; ring
    rw [this]
    exact div_nonneg (sq_nonneg _) (le_of_lt key)
