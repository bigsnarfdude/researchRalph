import Mathlib
set_option maxHeartbeats 2000000
open BigOperators Real Nat Topology Rat

theorem amc12a_2003_p24 :
  IsGreatest { y : ℝ | ∃ a b : ℝ, 1 < b ∧ b ≤ a ∧ y = Real.logb a (a / b) + Real.logb b (b / a) }
    0 := by
  constructor
  · simp only [Set.mem_setOf_eq]
    exact ⟨2, 2, by norm_num, le_refl _, by simp [Real.logb, Real.log_one]⟩
  · intro y hy
    simp only [Set.mem_setOf_eq] at hy
    obtain ⟨a, b, hb1, hba, hy⟩ := hy
    rw [hy]
    have ha1 : 1 < a := lt_of_lt_of_le hb1 hba
    have hb_pos : 0 < b := by linarith
    have ha_pos : 0 < a := by linarith
    have hlog_b_pos : 0 < Real.log b := Real.log_pos hb1
    have hlog_a_pos : 0 < Real.log a := Real.log_pos ha1
    rw [Real.logb, Real.logb, Real.log_div (ne_of_gt ha_pos) (ne_of_gt hb_pos),
        Real.log_div (ne_of_gt hb_pos) (ne_of_gt ha_pos)]
    have key : Real.log b / Real.log a + Real.log a / Real.log b ≥ 2 := by
      nlinarith [sq_nonneg (Real.log a / Real.log b - 1),
                 div_pos hlog_a_pos hlog_b_pos,
                 div_pos hlog_b_pos hlog_a_pos,
                 mul_div_cancel₀ (Real.log a) (ne_of_gt hlog_b_pos),
                 mul_div_cancel₀ (Real.log b) (ne_of_gt hlog_a_pos)]
    have goal_equiv : (Real.log a - Real.log b) / Real.log a +
        (Real.log b - Real.log a) / Real.log b =
        2 - (Real.log b / Real.log a + Real.log a / Real.log b) := by
      field_simp; ring
    linarith [goal_equiv]
