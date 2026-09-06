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

-- PHASE 2: Generalization to bases (3, 5)
-- Both bases give restricted sets, unlike (2,3)

def setA35 : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB35 : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}
def setAB35 : Set ℕ := {n | ∃ a ∈ setA35, ∃ b ∈ setB35, a + b = n}

private lemma setA35_le_40 {n : ℕ} (hn : n ∈ setA35) (hlt : n < 81) : n ≤ 40 := by
  simp only [setA35, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 81, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 40 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB35_le_31 {n : ℕ} (hn : n ∈ setB35) (hlt : n < 125) : n ≤ 31 := by
  simp only [setB35, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 125, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 31 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_35 : ∃ n : ℕ, n ∉ setAB35 := by
  use 72
  simp only [setAB35, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 40 := setA35_le_40 ha_A (by omega)
  have hb_bound : b ≤ 31 := setB35_le_31 hb_B (by omega)
  omega

-- PHASE 2: Generalization to bases (5, 6)
def setA56 : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}
def setB56 : Set ℕ := {n | ∀ d ∈ Nat.digits 6 n, d ≤ 1}
def setAB56 : Set ℕ := {n | ∃ a ∈ setA56, ∃ b ∈ setB56, a + b = n}

private lemma setA56_le_31 {n : ℕ} (hn : n ∈ setA56) (hlt : n < 125) : n ≤ 31 := by
  simp only [setA56, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 125, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 31 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB56_le_43 {n : ℕ} (hn : n ∈ setB56) (hlt : n < 216) : n ≤ 43 := by
  simp only [setB56, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 216, (∀ d ∈ Nat.digits 6 m, d ≤ 1) → m ≤ 43 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_56 : ∃ n : ℕ, n ∉ setAB56 := by
  use 75
  simp only [setAB56, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 31 := setA56_le_31 ha_A (by omega)
  have hb_bound : b ≤ 43 := setB56_le_43 hb_B (by omega)
  omega

-- PHASE 2: Generalization to bases (6, 9)
def setA69 : Set ℕ := {n | ∀ d ∈ Nat.digits 6 n, d ≤ 1}
def setB69 : Set ℕ := {n | ∀ d ∈ Nat.digits 9 n, d ≤ 1}
def setAB69 : Set ℕ := {n | ∃ a ∈ setA69, ∃ b ∈ setB69, a + b = n}

private lemma setA69_le_43 {n : ℕ} (hn : n ∈ setA69) (hlt : n < 216) : n ≤ 43 := by
  simp only [setA69, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 216, (∀ d ∈ Nat.digits 6 m, d ≤ 1) → m ≤ 43 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB69_le_121 {n : ℕ} (hn : n ∈ setB69) (hlt : n < 729) : n ≤ 121 := by
  simp only [setB69, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 729, (∀ d ∈ Nat.digits 9 m, d ≤ 1) → m ≤ 121 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_69 : ∃ n : ℕ, n ∉ setAB69 := by
  use 165
  simp only [setAB69, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 43 := setA69_le_43 ha_A (by omega)
  have hb_bound : b ≤ 121 := setB69_le_121 hb_B (by omega)
  omega

-- PHASE 2: Generalization to bases (7, 10)
def setA710 : Set ℕ := {n | ∀ d ∈ Nat.digits 7 n, d ≤ 1}
def setB710 : Set ℕ := {n | ∀ d ∈ Nat.digits 10 n, d ≤ 1}
def setAB710 : Set ℕ := {n | ∃ a ∈ setA710, ∃ b ∈ setB710, a + b = n}

private lemma setA710_le_57 {n : ℕ} (hn : n ∈ setA710) (hlt : n < 343) : n ≤ 57 := by
  simp only [setA710, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 343, (∀ d ∈ Nat.digits 7 m, d ≤ 1) → m ≤ 57 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB710_le_181 {n : ℕ} (hn : n ∈ setB710) (hlt : n < 1000) : n ≤ 181 := by
  simp only [setB710, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 1000, (∀ d ∈ Nat.digits 10 m, d ≤ 1) → m ≤ 181 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_710 : ∃ n : ℕ, n ∉ setAB710 := by
  use 239
  simp only [setAB710, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 57 := setA710_le_57 ha_A (by omega)
  have hb_bound : b ≤ 181 := setB710_le_181 hb_B (by omega)
  omega

-- PHASE 2: Generalization to bases (8, 11)
def setA811 : Set ℕ := {n | ∀ d ∈ Nat.digits 8 n, d ≤ 1}
def setB811 : Set ℕ := {n | ∀ d ∈ Nat.digits 11 n, d ≤ 1}
def setAB811 : Set ℕ := {n | ∃ a ∈ setA811, ∃ b ∈ setB811, a + b = n}

private lemma setA811_le_73 {n : ℕ} (hn : n ∈ setA811) (hlt : n < 512) : n ≤ 73 := by
  simp only [setA811, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 512, (∀ d ∈ Nat.digits 8 m, d ≤ 1) → m ≤ 73 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB811_le_265 {n : ℕ} (hn : n ∈ setB811) (hlt : n < 1331) : n ≤ 265 := by
  simp only [setB811, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 1331, (∀ d ∈ Nat.digits 11 m, d ≤ 1) → m ≤ 265 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_811 : ∃ n : ℕ, n ∉ setAB811 := by
  use 339
  simp only [setAB811, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 73 := setA811_le_73 ha_A (by omega)
  have hb_bound : b ≤ 265 := setB811_le_265 hb_B (by omega)
  omega

-- PHASE 2: Generalization to bases (9, 12)
def setA912 : Set ℕ := {n | ∀ d ∈ Nat.digits 9 n, d ≤ 1}
def setB912 : Set ℕ := {n | ∀ d ∈ Nat.digits 12 n, d ≤ 1}
def setAB912 : Set ℕ := {n | ∃ a ∈ setA912, ∃ b ∈ setB912, a + b = n}

private lemma setA912_le_121 {n : ℕ} (hn : n ∈ setA912) (hlt : n < 729) : n ≤ 121 := by
  simp only [setA912, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 729, (∀ d ∈ Nat.digits 9 m, d ≤ 1) → m ≤ 121 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB912_le_157 {n : ℕ} (hn : n ∈ setB912) (hlt : n < 1728) : n ≤ 157 := by
  simp only [setB912, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 1728, (∀ d ∈ Nat.digits 12 m, d ≤ 1) → m ≤ 157 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_912 : ∃ n : ℕ, n ∉ setAB912 := by
  use 279
  simp only [setAB912, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 121 := setA912_le_121 ha_A (by omega)
  have hb_bound : b ≤ 157 := setB912_le_157 hb_B (by omega)
  omega

-- PHASE 2: Generalization to bases (10, 13)
def setA1013 : Set ℕ := {n | ∀ d ∈ Nat.digits 10 n, d ≤ 1}
def setB1013 : Set ℕ := {n | ∀ d ∈ Nat.digits 13 n, d ≤ 1}
def setAB1013 : Set ℕ := {n | ∃ a ∈ setA1013, ∃ b ∈ setB1013, a + b = n}

private lemma setA1013_le_181 {n : ℕ} (hn : n ∈ setA1013) (hlt : n < 1000) : n ≤ 181 := by
  simp only [setA1013, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 1000, (∀ d ∈ Nat.digits 10 m, d ≤ 1) → m ≤ 181 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB1013_le_183 {n : ℕ} (hn : n ∈ setB1013) (hlt : n < 2197) : n ≤ 183 := by
  simp only [setB1013, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 2197, (∀ d ∈ Nat.digits 13 m, d ≤ 1) → m ≤ 183 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_1013 : ∃ n : ℕ, n ∉ setAB1013 := by
  use 365
  simp only [setAB1013, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 181 := setA1013_le_181 ha_A (by omega)
  have hb_bound : b ≤ 183 := setB1013_le_183 hb_B (by omega)
  omega

-- PHASE 2: Generalization to bases (11, 14)
def setA1114 : Set ℕ := {n | ∀ d ∈ Nat.digits 11 n, d ≤ 1}
def setB1114 : Set ℕ := {n | ∀ d ∈ Nat.digits 14 n, d ≤ 1}
def setAB1114 : Set ℕ := {n | ∃ a ∈ setA1114, ∃ b ∈ setB1114, a + b = n}

private lemma setA1114_le_265 {n : ℕ} (hn : n ∈ setA1114) (hlt : n < 1331) : n ≤ 265 := by
  simp only [setA1114, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 1331, (∀ d ∈ Nat.digits 11 m, d ≤ 1) → m ≤ 265 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB1114_le_211 {n : ℕ} (hn : n ∈ setB1114) (hlt : n < 2744) : n ≤ 211 := by
  simp only [setB1114, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 2744, (∀ d ∈ Nat.digits 14 m, d ≤ 1) → m ≤ 211 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_1114 : ∃ n : ℕ, n ∉ setAB1114 := by
  use 477
  simp only [setAB1114, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 265 := setA1114_le_265 ha_A (by omega)
  have hb_bound : b ≤ 211 := setB1114_le_211 hb_B (by omega)
  omega
