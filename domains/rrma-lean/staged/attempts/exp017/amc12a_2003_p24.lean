import Mathlib

set_option maxHeartbeats 16000000

open BigOperators Real Nat Topology Rat

-- Max of logb a (a/b) + logb b (b/a) = 0 (AM-GM on log ratio)
theorem amc12a_2003_p24 :
  IsGreatest { y : ℝ | ∃ a b : ℝ, 1 < b ∧ b ≤ a ∧ y = Real.logb a (a / b) + Real.logb b (b / a) }
    0 := by
  constructor
  · -- 0 ∈ S: take a = b = 2
    refine ⟨2, 2, by norm_num, le_refl _, ?_⟩
    simp [Real.logb, div_self (ne_of_gt (Real.log_pos (by norm_num : (1:ℝ) < 2)))]
  · -- ∀ y ∈ S, y ≤ 0
    intro y ⟨a, b, hb, hab, hy⟩
    rw [hy]
    have hb1 : 1 < b := hb
    have ha1 : 1 < a := lt_of_lt_of_le hb hab
    have hlogb : 0 < Real.log b := Real.log_pos hb1
    have hloga : 0 < Real.log a := Real.log_pos ha1
    have ha_pos : 0 < a := by linarith
    have hb_pos : 0 < b := by linarith
    -- logb a (a/b) = log(a/b)/log(a) = (log a - log b)/log a = 1 - log b/log a
    -- logb b (b/a) = log(b/a)/log(b) = (log b - log a)/log b = 1 - log a/log b
    -- Sum = 2 - log b/log a - log a/log b
    have h_sum : Real.logb a (a / b) + Real.logb b (b / a) =
        2 - Real.log b / Real.log a - Real.log a / Real.log b := by
      simp only [Real.logb, Real.log_div (ne_of_gt ha_pos) (ne_of_gt hb_pos),
                  Real.log_div (ne_of_gt hb_pos) (ne_of_gt ha_pos)]
      field_simp
      ring
    rw [h_sum]
    -- 2 - x - 1/x ≤ 0 for x > 0 (where x = log a / log b ≥ 1)
    -- ⟺ x + 1/x ≥ 2 (AM-GM)
    linarith [div_add_div_same (Real.log b) (Real.log a) (Real.log a),
              sq_nonneg (Real.log a / Real.log b - 1),
              div_pos hloga hlogb, div_pos hlogb hloga]
