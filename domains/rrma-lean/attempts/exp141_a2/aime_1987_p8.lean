import Mathlib
set_option maxHeartbeats 64000000
set_option linter.unusedVariables false
set_option linter.unusedTactic false
open BigOperators Real Nat Topology Rat

theorem aime_1987_p8 :
  IsGreatest { n : ℕ | 0 < n ∧ ∃! k : ℕ, (8 : ℝ) / 15 < n / (n + k) ∧ (n : ℝ) / (n + k) < 7 / 13 } 112 := by
  constructor
  · simp only [Set.mem_setOf_eq]
    refine ⟨by norm_num, 97, ⟨by norm_num, by norm_num⟩, ?_⟩
    intro k ⟨hlo, hhi⟩
    have hpos : (0 : ℝ) < ↑112 + ↑k := by positivity
    push_cast at hlo hhi
    have h1 : 8 * ((112 : ℝ) + ↑k) < 15 * 112 := by
      rw [div_lt_div_iff₀ (by positivity : (0:ℝ) < 15) (by positivity)] at hlo; linarith
    have h2 : 13 * (112 : ℝ) < 7 * (112 + ↑k) := by
      rw [div_lt_div_iff₀ (by positivity) (by positivity : (0:ℝ) < 13)] at hhi; linarith
    have h1n : 8 * (112 + k) < 15 * 112 := by exact_mod_cast h1
    have h2n : 13 * 112 < 7 * (112 + k) := by exact_mod_cast h2
    omega
  · intro m hm
    simp only [Set.mem_setOf_eq] at hm
    obtain ⟨hpos_m, k₀, ⟨hlo, hhi⟩, huniq⟩ := hm
    by_contra hgt; push_neg at hgt
    have hpos : (0 : ℝ) < ↑m + ↑k₀ := by positivity
    push_cast at hlo hhi
    have h1r : 8 * ((m : ℝ) + ↑k₀) < 15 * ↑m := by
      rw [div_lt_div_iff₀ (by positivity : (0:ℝ) < 15) (by positivity)] at hlo; linarith
    have h2r : 13 * (m : ℝ) < 7 * (↑m + ↑k₀) := by
      rw [div_lt_div_iff₀ (by positivity) (by positivity : (0:ℝ) < 13)] at hhi; linarith
    set k₁ := 6 * m / 7 + 1
    -- k₁ and k₁+1 both satisfy the ℕ cross-multiplication bounds for m ≥ 113
    have hk1_hi : 8 * (m + k₁) < 15 * m := by omega
    have hk1_lo : 13 * m < 7 * (m + k₁) := by omega
    have hk2_hi : 8 * (m + (k₁ + 1)) < 15 * m := by omega
    have hk2_lo : 13 * m < 7 * (m + (k₁ + 1)) := by omega
    -- Convert to ℝ fraction bounds via cast + div_lt_div_iff₀
    have hk1_frac_lo : (8 : ℝ) / 15 < ↑m / (↑m + ↑k₁) := by
      rw [div_lt_div_iff₀ (by positivity : (0:ℝ) < 15) (by positivity : (0:ℝ) < ↑m + ↑k₁)]
      have : (8 : ℝ) * (↑m + ↑k₁) < 15 * ↑m := by exact_mod_cast hk1_hi
      linarith
    have hk1_frac_hi : (↑m : ℝ) / (↑m + ↑k₁) < 7 / 13 := by
      rw [div_lt_div_iff₀ (by positivity : (0:ℝ) < ↑m + ↑k₁) (by positivity : (0:ℝ) < 13)]
      have : (13 : ℝ) * ↑m < 7 * (↑m + ↑k₁) := by exact_mod_cast hk1_lo
      linarith
    have hk2_frac_lo : (8 : ℝ) / 15 < ↑m / (↑m + ↑(k₁ + 1)) := by
      rw [div_lt_div_iff₀ (by positivity : (0:ℝ) < 15) (by positivity : (0:ℝ) < ↑m + ↑(k₁+1))]
      have : (8 : ℝ) * (↑m + ↑(k₁+1)) < 15 * ↑m := by exact_mod_cast hk2_hi
      linarith
    have hk2_frac_hi : (↑m : ℝ) / (↑m + ↑(k₁ + 1)) < 7 / 13 := by
      rw [div_lt_div_iff₀ (by positivity : (0:ℝ) < ↑m + ↑(k₁+1)) (by positivity : (0:ℝ) < 13)]
      have : (13 : ℝ) * ↑m < 7 * (↑m + ↑(k₁+1)) := by exact_mod_cast hk2_lo
      linarith
    have := huniq k₁ ⟨hk1_frac_lo, hk1_frac_hi⟩
    have := huniq (k₁ + 1) ⟨hk2_frac_lo, hk2_frac_hi⟩
    omega
