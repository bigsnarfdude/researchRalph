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

-- Phase 2: base pair (3,7)
-- max(setA ∩ [0,27)) = 13, max(setG ∩ [0,49)) = 8. 13+8=21 < 22 ≤ min(27,49)=27, so n=22 is a gap.
def setG : Set ℕ := {n | ∀ d ∈ Nat.digits 7 n, d ≤ 1}
def setAG : Set ℕ := {n | ∃ a ∈ setA, ∃ g ∈ setG, a + g = n}

private lemma setG_le_8 {n : ℕ} (hn : n ∈ setG) (hlt : n < 49) : n ≤ 8 := by
  simp only [setG, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 49, (∀ d ∈ Nat.digits 7 m, d ≤ 1) → m ≤ 8 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_3_7 : ∃ n : ℕ, n ∉ setAG := by
  use 22
  simp only [setAG, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, g, hg_G, hag⟩
  have ha_bound : a ≤ 13 := setA_le_13 ha_A (by omega)
  have hg_bound : g ≤ 8 := setG_le_8 hg_G (by omega)
  omega

theorem erdos_125_generalized_3_7 : ∃ n : ℕ, n ∉ setAG :=
  gap_exists_3_7

-- Phase 2: base pair (4,5)
-- max(setF ∩ [0,16)) = 5, max(setE ∩ [0,25)) = 6. 5+6=11 < 12 ≤ min(16,25)=16, so n=12 is a gap.
def setF : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setE : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}
def setFE : Set ℕ := {n | ∃ f ∈ setF, ∃ e ∈ setE, f + e = n}

private lemma setF_le_5 {n : ℕ} (hn : n ∈ setF) (hlt : n < 16) : n ≤ 5 := by
  simp only [setF, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 16, (∀ d ∈ Nat.digits 4 m, d ≤ 1) → m ≤ 5 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setE_le_6 {n : ℕ} (hn : n ∈ setE) (hlt : n < 25) : n ≤ 6 := by
  simp only [setE, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 25, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 6 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_4_5 : ∃ n : ℕ, n ∉ setFE := by
  use 12
  simp only [setFE, Set.mem_setOf_eq]
  rintro ⟨f, hf_F, e, he_E, hfe⟩
  have hf_bound : f ≤ 5 := setF_le_5 hf_F (by omega)
  have he_bound : e ≤ 6 := setE_le_6 he_E (by omega)
  omega

theorem erdos_125_generalized_4_5 : ∃ n : ℕ, n ∉ setFE :=
  gap_exists_4_5

-- Phase 2: base pair (5,7)
-- max(setE ∩ [0,25)) = 6, max(setG ∩ [0,49)) = 8. 6+8=14 < 15 ≤ min(25,49)=25, so n=15 is a gap.
def setEG : Set ℕ := {n | ∃ e ∈ setE, ∃ g ∈ setG, e + g = n}

lemma gap_exists_5_7 : ∃ n : ℕ, n ∉ setEG := by
  use 15
  simp only [setEG, Set.mem_setOf_eq]
  rintro ⟨e, he_E, g, hg_G, heg⟩
  have he_bound : e ≤ 6 := setE_le_6 he_E (by omega)
  have hg_bound : g ≤ 8 := setG_le_8 hg_G (by omega)
  omega

theorem erdos_125_generalized_5_7 : ∃ n : ℕ, n ∉ setEG :=
  gap_exists_5_7

-- Phase 2: base pair (3,8)
-- max(setA ∩ [0,27)) = 13, max(setH ∩ [0,64)) = 9. 13+9=22 < 23 ≤ min(27,64)=27, so n=23 is a gap.
def setH : Set ℕ := {n | ∀ d ∈ Nat.digits 8 n, d ≤ 1}
def setAH : Set ℕ := {n | ∃ a ∈ setA, ∃ h ∈ setH, a + h = n}

private lemma setH_le_9 {n : ℕ} (hn : n ∈ setH) (hlt : n < 64) : n ≤ 9 := by
  simp only [setH, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 64, (∀ d ∈ Nat.digits 8 m, d ≤ 1) → m ≤ 9 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_3_8 : ∃ n : ℕ, n ∉ setAH := by
  use 23
  simp only [setAH, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, h, hh_H, hah⟩
  have ha_bound : a ≤ 13 := setA_le_13 ha_A (by omega)
  have hh_bound : h ≤ 9 := setH_le_9 hh_H (by omega)
  omega

theorem erdos_125_generalized_3_8 : ∃ n : ℕ, n ∉ setAH :=
  gap_exists_3_8

-- Phase 2: base pair (4,7)
-- max(setF ∩ [0,16)) = 5, max(setG ∩ [0,49)) = 8. 5+8=13 < 14 ≤ min(16,49)=16, so n=14 is a gap.
def setFG : Set ℕ := {n | ∃ f ∈ setF, ∃ g ∈ setG, f + g = n}

lemma gap_exists_4_7 : ∃ n : ℕ, n ∉ setFG := by
  use 14
  simp only [setFG, Set.mem_setOf_eq]
  rintro ⟨f, hf_F, g, hg_G, hfg⟩
  have hf_bound : f ≤ 5 := setF_le_5 hf_F (by omega)
  have hg_bound : g ≤ 8 := setG_le_8 hg_G (by omega)
  omega

theorem erdos_125_generalized_4_7 : ∃ n : ℕ, n ∉ setFG :=
  gap_exists_4_7

-- Phase 2: base pair (5,8)
-- max(setE ∩ [0,25)) = 6, max(setH ∩ [0,64)) = 9. 6+9=15 < 16 ≤ min(25,64)=25, so n=16 is a gap.
def setEH : Set ℕ := {n | ∃ e ∈ setE, ∃ h ∈ setH, e + h = n}

lemma gap_exists_5_8 : ∃ n : ℕ, n ∉ setEH := by
  use 16
  simp only [setEH, Set.mem_setOf_eq]
  rintro ⟨e, he_E, h, hh_H, heh⟩
  have he_bound : e ≤ 6 := setE_le_6 he_E (by omega)
  have hh_bound : h ≤ 9 := setH_le_9 hh_H (by omega)
  omega

theorem erdos_125_generalized_5_8 : ∃ n : ℕ, n ∉ setEH :=
  gap_exists_5_8


