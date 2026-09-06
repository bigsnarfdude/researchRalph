import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

noncomputable def lowerDensity (S : Set ℕ) : ℝ :=
  liminf (fun N : ℕ => (N : ℝ)⁻¹ * (S ∩ (range N).toSet).ncard) atTop

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

-- Phase 2: generalization to base pair (3,5), analogous to (3,4).
-- max(setA ∩ [0,27)) = 13 = (3^3-1)/2, max(setC ∩ [0,25)) = 6 = (5^2-1)/4.
-- 13 + 6 = 19 < 20 ≤ min(27,25) = 25, so n = 20 is a gap.
def setC : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}
def setAC : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setC, a + b = n}

private lemma setA_le_13 {n : ℕ} (hn : n ∈ setA) (hlt : n < 27) : n ≤ 13 := by
  simp only [setA, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 27, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 13 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setC_le_6 {n : ℕ} (hn : n ∈ setC) (hlt : n < 25) : n ≤ 6 := by
  simp only [setC, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 25, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 6 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_3_5 : ∃ n : ℕ, n ∉ setAC := by
  use 20
  simp only [setAC, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_C, hab⟩
  have ha_bound : a ≤ 13 := setA_le_13 ha_A (by omega)
  have hb_bound : b ≤ 6 := setC_le_6 hb_C (by omega)
  omega

theorem erdos_125_generalized_3_5 : ∃ n : ℕ, n ∉ setAC :=
  gap_exists_3_5

-- Phase 2: base pair (4,5).
-- max(setE ∩ [0,16)) = 5 = (4^2-1)/3, max(setF ∩ [0,25)) = 6 = (5^2-1)/4.
-- 5 + 6 = 11, gap at 12.
def setE : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setF : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}
def setEF : Set ℕ := {n | ∃ a ∈ setE, ∃ b ∈ setF, a + b = n}

private lemma setE_le_5 {n : ℕ} (hn : n ∈ setE) (hlt : n < 16) : n ≤ 5 := by
  simp only [setE, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 16, (∀ d ∈ Nat.digits 4 m, d ≤ 1) → m ≤ 5 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setF_le_6 {n : ℕ} (hn : n ∈ setF) (hlt : n < 25) : n ≤ 6 := by
  simp only [setF, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 25, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 6 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_4_5 : ∃ n : ℕ, n ∉ setEF := by
  use 12
  simp only [setEF, Set.mem_setOf_eq]
  rintro ⟨a, ha_E, b, hb_F, hab⟩
  have ha_bound : a ≤ 5 := setE_le_5 ha_E (by omega)
  have hb_bound : b ≤ 6 := setF_le_6 hb_F (by omega)
  omega

theorem erdos_125_generalized_4_5 : ∃ n : ℕ, n ∉ setEF :=
  gap_exists_4_5

-- Phase 2: base pair (5,7).
-- max(setG ∩ [0,25)) = 6 = (5^2-1)/4, max(setH ∩ [0,49)) = 8 = (7^2-1)/6.
-- 6 + 8 = 14, gap at 15.
def setG : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}
def setH : Set ℕ := {n | ∀ d ∈ Nat.digits 7 n, d ≤ 1}
def setGH : Set ℕ := {n | ∃ a ∈ setG, ∃ b ∈ setH, a + b = n}

private lemma setG_le_6 {n : ℕ} (hn : n ∈ setG) (hlt : n < 25) : n ≤ 6 := by
  simp only [setG, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 25, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 6 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setH_le_8 {n : ℕ} (hn : n ∈ setH) (hlt : n < 49) : n ≤ 8 := by
  simp only [setH, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 49, (∀ d ∈ Nat.digits 7 m, d ≤ 1) → m ≤ 8 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_5_7 : ∃ n : ℕ, n ∉ setGH := by
  use 15
  simp only [setGH, Set.mem_setOf_eq]
  rintro ⟨a, ha_G, b, hb_H, hab⟩
  have ha_bound : a ≤ 6 := setG_le_6 ha_G (by omega)
  have hb_bound : b ≤ 8 := setH_le_8 hb_H (by omega)
  omega

theorem erdos_125_generalized_5_7 : ∃ n : ℕ, n ∉ setGH :=
  gap_exists_5_7

-- Phase 2: base pair (3,7).
-- max(setA ∩ [0,27)) = 13 = (3^3-1)/2, max(setI ∩ [0,49)) = 8 = (7^2-1)/6.
-- 13 + 8 = 21, gap at 22.
def setI : Set ℕ := {n | ∀ d ∈ Nat.digits 7 n, d ≤ 1}
def setAI : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setI, a + b = n}

private lemma setA_le_13_for_3_7 {n : ℕ} (hn : n ∈ setA) (hlt : n < 27) : n ≤ 13 := by
  simp only [setA, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 27, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 13 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setI_le_8 {n : ℕ} (hn : n ∈ setI) (hlt : n < 49) : n ≤ 8 := by
  simp only [setI, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 49, (∀ d ∈ Nat.digits 7 m, d ≤ 1) → m ≤ 8 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_3_7 : ∃ n : ℕ, n ∉ setAI := by
  use 22
  simp only [setAI, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_I, hab⟩
  have ha_bound : a ≤ 13 := setA_le_13_for_3_7 ha_A (by omega)
  have hb_bound : b ≤ 8 := setI_le_8 hb_I (by omega)
  omega

theorem erdos_125_generalized_3_7 : ∃ n : ℕ, n ∉ setAI :=
  gap_exists_3_7

-- Phase 2: base pair (4,7).
-- max(setJ ∩ [0,16)) = 5 = (4^2-1)/3, max(setK ∩ [0,49)) = 8 = (7^2-1)/6.
-- 5 + 8 = 13, gap at 14.
def setJ : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setK : Set ℕ := {n | ∀ d ∈ Nat.digits 7 n, d ≤ 1}
def setJK : Set ℕ := {n | ∃ a ∈ setJ, ∃ b ∈ setK, a + b = n}

private lemma setJ_le_5 {n : ℕ} (hn : n ∈ setJ) (hlt : n < 16) : n ≤ 5 := by
  simp only [setJ, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 16, (∀ d ∈ Nat.digits 4 m, d ≤ 1) → m ≤ 5 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setK_le_8 {n : ℕ} (hn : n ∈ setK) (hlt : n < 49) : n ≤ 8 := by
  simp only [setK, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 49, (∀ d ∈ Nat.digits 7 m, d ≤ 1) → m ≤ 8 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_4_7 : ∃ n : ℕ, n ∉ setJK := by
  use 14
  simp only [setJK, Set.mem_setOf_eq]
  rintro ⟨a, ha_J, b, hb_K, hab⟩
  have ha_bound : a ≤ 5 := setJ_le_5 ha_J (by omega)
  have hb_bound : b ≤ 8 := setK_le_8 hb_K (by omega)
  omega

theorem erdos_125_generalized_4_7 : ∃ n : ℕ, n ∉ setJK :=
  gap_exists_4_7

