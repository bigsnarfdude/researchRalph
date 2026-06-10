import Mathlib

set_option maxHeartbeats 128000000
set_option maxRecDepth 4096
set_option linter.all false

open BigOperators Real Nat Topology Rat

theorem aime_1987_p8 :
  IsGreatest { n : ℕ | 0 < n ∧ ∃! k : ℕ, (8 : ℝ) / 15 < n / (n + k) ∧ (n : ℝ) / (n + k) < 7 / 13 } 112 := by
  refine ⟨⟨by norm_num, 97, ⟨by norm_num, by norm_num⟩, ?_⟩, ?_⟩
  · intro k ⟨h1, h2⟩
    have hpos : (0:ℝ) < (112:ℕ) + ↑k := by positivity
    have h1' := (div_lt_div_iff₀ (by norm_num : (0:ℝ) < 15) hpos).mp h1
    have h2' := (div_lt_div_iff₀ hpos (by norm_num : (0:ℝ) < 13)).mp h2
    have h1n : 8 * (112 + k) < 112 * 15 := by exact_mod_cast h1'
    have h2n : 112 * 13 < 7 * (112 + k) := by exact_mod_cast h2'
    omega
  · rintro m ⟨hm_pos, k, ⟨hk1, hk2⟩, huniq⟩
    by_contra hgt; push_neg at hgt
    have hm : 113 ≤ m := by omega
    have hpos : (0:ℝ) < ↑m + ↑k := by positivity
    set k₁ := 6*m/7 + 1
    set k₂ := 6*m/7 + 2
    have g1 : 6*m < 7*k₁ := by omega
    have g2 : 8*k₁ < 7*m := by omega
    have g3 : 6*m < 7*k₂ := by omega
    have g4 : 8*k₂ < 7*m := by omega
    have hpos1 : (0:ℝ) < ↑m + ↑k₁ := by positivity
    have hpos2 : (0:ℝ) < ↑m + ↑k₂ := by positivity
    have c1 : (8:ℝ)/15 < ↑m/(↑m+↑k₁) := by
      rw [div_lt_div_iff₀ (by norm_num : (0:ℝ) < 15) hpos1]
      have h : 8*(m+k₁) < m*15 := by omega
      exact_mod_cast h
    have c2 : (↑m:ℝ)/(↑m+↑k₁) < 7/13 := by
      rw [div_lt_div_iff₀ hpos1 (by norm_num : (0:ℝ) < 13)]
      have h : m*13 < 7*(m+k₁) := by omega
      exact_mod_cast h
    have c3 : (8:ℝ)/15 < ↑m/(↑m+↑k₂) := by
      rw [div_lt_div_iff₀ (by norm_num : (0:ℝ) < 15) hpos2]
      have h : 8*(m+k₂) < m*15 := by omega
      exact_mod_cast h
    have c4 : (↑m:ℝ)/(↑m+↑k₂) < 7/13 := by
      rw [div_lt_div_iff₀ hpos2 (by norm_num : (0:ℝ) < 13)]
      have h : m*13 < 7*(m+k₂) := by omega
      exact_mod_cast h
    have eq1 := huniq k₁ ⟨c1, c2⟩
    have eq2 := huniq k₂ ⟨c3, c4⟩
    omega
