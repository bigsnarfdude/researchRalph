import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat
theorem imo_1974_p5 (a b c d s : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h₁ : s = a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) :
  1 < s ∧ s < 2 := by
  have ha := h₀.1; have hb := h₀.2.1; have hc := h₀.2.2.1; have hd := h₀.2.2.2
  have h_abd : 0 < a + b + d := by linarith
  have h_abc : 0 < a + b + c := by linarith
  have h_bcd : 0 < b + c + d := by linarith
  have h_acd : 0 < a + c + d := by linarith
  rw [h₁]; constructor
  · -- s > 1: each x/(x+y+z) > x/(a+b+c+d) since x+y+z < a+b+c+d
    have h1 : a / (a+b+d) > a / (a+b+c+d) := by
      exact div_lt_div_of_pos_left ha (by linarith) (by linarith)
    have h2 : b / (a+b+c) > b / (a+b+c+d) := by
      exact div_lt_div_of_pos_left hb (by linarith) (by linarith)
    have h3 : c / (b+c+d) > c / (a+b+c+d) := by
      exact div_lt_div_of_pos_left hc (by linarith) (by linarith)
    have h4 : d / (a+c+d) > d / (a+b+c+d) := by
      exact div_lt_div_of_pos_left hd (by linarith) (by linarith)
    have h_sum : a/(a+b+c+d) + b/(a+b+c+d) + c/(a+b+c+d) + d/(a+b+c+d) = 1 := by
      field_simp
    linarith
  · -- s < 2: pair (a,d) and (b,c)
    -- a/(a+b+d) + d/(a+c+d) < 1 because Den - Num = ab + bc + cd > 0
    have hp1 : a / (a+b+d) + d / (a+c+d) < 1 := by
      rw [div_add_div _ _ (ne_of_gt h_abd) (ne_of_gt h_acd)]
      rw [div_lt_one (mul_pos h_abd h_acd)]
      nlinarith
    -- b/(a+b+c) + c/(b+c+d) < 1 because Den - Num = ad + cd + ab > 0... let me check
    have hp2 : b / (a+b+c) + c / (b+c+d) < 1 := by
      rw [div_add_div _ _ (ne_of_gt h_abc) (ne_of_gt h_bcd)]
      rw [div_lt_one (mul_pos h_abc h_bcd)]
      nlinarith
    linarith
