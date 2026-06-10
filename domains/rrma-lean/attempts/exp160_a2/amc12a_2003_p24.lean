import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat

theorem amc12a_2003_p24 :
  IsGreatest { y : ℝ | ∃ a b : ℝ, 1 < b ∧ b ≤ a ∧ y = Real.logb a (a / b) + Real.logb b (b / a) }
    0 := by
  constructor
  · -- 0 ∈ S: take a = b = 2
    refine ⟨2, 2, by norm_num, le_refl 2, ?_⟩
    simp [Real.logb]
  · -- ∀ y ∈ S, y ≤ 0
    intro y ⟨a, b, hb1, hab, hy⟩
    rw [hy]
    have ha1 : 1 < a := lt_of_lt_of_le hb1 hab
    have hb0 : (0:ℝ) < b := by linarith
    have ha0 : (0:ℝ) < a := by linarith
    have hlb : 0 < Real.log b := Real.log_pos hb1
    have hla : 0 < Real.log a := Real.log_pos ha1
    rw [Real.logb, Real.logb]
    rw [Real.log_div (ne_of_gt ha0) (ne_of_gt hb0)]
    rw [Real.log_div (ne_of_gt hb0) (ne_of_gt ha0)]
    rw [div_add_div _ _ (ne_of_gt hla) (ne_of_gt hlb)]
    apply div_nonpos_of_nonpos_of_nonneg
    · nlinarith [sq_nonneg (Real.log a - Real.log b)]
    · exact le_of_lt (mul_pos hla hlb)
