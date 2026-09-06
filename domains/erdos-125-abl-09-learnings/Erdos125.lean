import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

noncomputable def lowerDensity (S : Set ℕ) : ℝ :=
  liminf (fun N : ℕ => (N : ℝ)⁻¹ * (S ∩ (range N).toSet).ncard) atTop

lemma exists_k_m_ratio_close (ε : ℝ) (hε : 0 < ε) :
    ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ |↑k * log 3 - ↑m * log 4| < ε := by
  have hlog3_pos : (0 : ℝ) < log 3 := Real.log_pos (by norm_num)
  have hlog4_pos : (0 : ℝ) < log 4 := Real.log_pos (by norm_num)
  obtain ⟨N, hN⟩ := exists_nat_gt (log 4 / ε)
  have hN_pos : 0 < N + 1 := Nat.succ_pos _
  obtain ⟨j, k, hk_pos, hk_le, hbound⟩ :=
    Real.exists_int_int_abs_mul_sub_le (log 3 / log 4) hN_pos
  have hN2_bound : (1 : ℝ) / (↑(N + 1) + 1) < ε / log 4 := by
    have h_pos : (0 : ℝ) < ↑(N + 1) + 1 := by positivity
    have hNε : log 4 < (N : ℝ) * ε := by
      rw [div_lt_iff₀ hε] at hN; linarith
    rw [div_lt_iff₀ h_pos, div_mul_eq_mul_div, lt_div_iff₀ hlog4_pos]
    push_cast; linarith
  have hj_pos : 0 < j := by
    have hξ_pos : 0 < log 3 / log 4 := div_pos hlog3_pos hlog4_pos
    have hξ_gt_half : (1 : ℝ) / 2 < log 3 / log 4 := by
      rw [lt_div_iff₀ hlog4_pos]
      have h1 : log 4 < log 9 := Real.log_lt_log (by norm_num) (by norm_num)
      have h2 : log (9 : ℝ) = 2 * log 3 := by
        have h9 : (9 : ℝ) = 3 ^ 2 := by norm_num
        rw [h9, Real.log_pow]; push_cast; ring
      linarith
    have hk_real : (1 : ℝ) ≤ (k : ℝ) := by exact_mod_cast hk_pos
    have hkξ_gt_half : (1 : ℝ) / 2 < (k : ℝ) * (log 3 / log 4) := by
      nlinarith [mul_nonneg (show (0:ℝ) ≤ (k:ℝ) - 1 by linarith) hξ_pos.le]
    have h_half : (1 : ℝ) / (↑(N + 1) + 1) ≤ 1 / 2 := by
      have h2le : (2 : ℝ) ≤ ↑(N + 1) + 1 := by norm_cast; omega
      exact one_div_le_one_div_of_le (by norm_num) h2le
    have h_j_lower : (k : ℝ) * (log 3 / log 4) - (1 / (↑(N + 1) + 1)) ≤ (j : ℝ) := by
      have := (abs_le.mp hbound).2; linarith
    have hj_real : (0 : ℝ) < (j : ℝ) := by linarith
    exact_mod_cast hj_real
  refine ⟨k.toNat, j.toNat, by omega, by omega, ?_⟩
  have hk_cast : (k.toNat : ℝ) = (k : ℝ) := by
    exact_mod_cast Int.toNat_of_nonneg hk_pos.le
  have hj_cast : (j.toNat : ℝ) = (j : ℝ) := by
    exact_mod_cast Int.toNat_of_nonneg hj_pos.le
  rw [hk_cast, hj_cast]
  have h_rearrange : (k : ℝ) * log 3 - (j : ℝ) * log 4 =
      log 4 * ((k : ℝ) * (log 3 / log 4) - (j : ℝ)) := by
    field_simp
  rw [h_rearrange, abs_mul, abs_of_pos hlog4_pos]
  calc log 4 * |(k : ℝ) * (log 3 / log 4) - (j : ℝ)|
      ≤ log 4 * (1 / (↑(N + 1) + 1)) :=
        mul_le_mul_of_nonneg_left hbound hlog4_pos.le
    _ < log 4 * (ε / log 4) := mul_lt_mul_of_pos_left hN2_bound hlog4_pos
    _ = ε := by field_simp

private lemma setA_le_40 {n : ℕ} (hn : n ∈ setA) (hlt : n < 81) : n ≤ 40 := by
  simp only [setA, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 81, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 40 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB_le_21 {n : ℕ} (hn : n ∈ setB) (hlt : n < 64) : n ≤ 21 := by
  simp only [setB, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 64, (∀ d ∈ Nat.digits 4 m, d ≤ 1) → m ≤ 21 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_at_aligned_scale (k m : ℕ) (hk : 0 < k) (hm : 0 < m)
    (h_close : |↑k * log 3 - ↑m * log 4| < 1) :
    ∃ start width : ℕ, 0 < width ∧
    ∀ n ∈ Ico start (start + width), n ∉ setAB := by
  refine ⟨62, 2, by norm_num, fun n hn hn_ab => ?_⟩
  simp only [Finset.mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  simp only [setAB, Set.mem_setOf_eq] at hn_ab
  obtain ⟨a, ha_A, b, hb_B, hab⟩ := hn_ab
  have ha_bound : a ≤ 40 := setA_le_40 ha_A (by omega)
  have hb_bound : b ≤ 21 := setB_le_21 hb_B (by omega)
  omega

lemma gap_exists : ∃ n : ℕ, n ∉ setAB := by
  use 62
  simp only [setAB, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 40 := setA_le_40 ha_A (by omega)
  have hb_bound : b ≤ 21 := setB_le_21 hb_B (by omega)
  omega

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB :=
  gap_exists
